import math
import collections
from itertools import repeat
from typing import List, Union, Tuple, Optional

from scipy import ndimage
import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


def _reverse_repeat_tuple(t, n):
    r"""Reverse the order of `t` and repeat each element for `n` times.

    This can be used to translate padding arg used by Conv and Pooling modules
    to the ones used by `F.pad`.
    """
    return tuple(x for x in reversed(t) for _ in range(n))


class _SRE_ConvNd(nn.Module):
    r"""
    Official implementation of Symmetric Rotation Invariant Conv2d in PyTorch.
    Apply symmetric rotation invariant convolution over an input signal composed of several input planes.

    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        transposed (bool, optional): If ``True``, use a transposed convolution operator. Default: ``False``
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
        padding_mode (str, optional): Accepted values `zeros`, `reflect`, `replicate`, `circular`. Default: `zeros`
        sre_k (int, optional): Number of bands in the SRE kernel. If None, calculate based on kernel size. Default: None
        inference_accelerate (bool, optional): If ``True``, accelerate inference by pre-computing the weight matrix. Default: ``True``
        force_circular (bool, optional): If ``True``, force the kernel to be circular by remove the outmost corner pixel in the kernel. Default: ``True``
        device (torch.device, optional): Device to use. Default: ``None``
        dtype (torch.dtype, optional): Data type to use. Default: ``None``

    Attributes:
        weight (torch.Tensor): the learnable weights of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{ri\_k})`
        bias (torch.Tensor):   the learnable bias of the module of shape
            :math:`(\text{out\_channels})`
        weight_index_mat (torch.Tensor): the learnable index matrix of the module of shape
            :math:`(\text{in\_channels}, \text{out\_channels}, \text{ri\_k}, \text{ri\_k})`
        infer_weight_matrix (torch.Tensor): the pre-computed weight matrix of the module of shape
            :math:`(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{H}, \text{W})`
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, ...],
        stride: Tuple[int, ...] = 1,
        padding: Union[str, Tuple[int, ...]] = 0,
        dilation: Tuple[int, ...] = 1,
        transposed: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        sre_k: int = None,
        inference_accelerate: bool = True,
        force_circular: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        # TODO: this part goes to individual conv
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        valid_padding_strings = {"same", "valid"}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings
                    )
                )
            if padding == "same" and any(s != 1 for s in stride):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )
        valid_padding_modes = {"zeros", "reflect", "replicate", "circular"}
        if padding_mode not in valid_padding_modes:
            raise ValueError(
                "padding_mode must be one of {}, but got padding_mode='{}'".format(
                    valid_padding_modes, padding_mode
                )
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.groups = groups
        self.padding_mode = padding_mode
        self.output_padding = self.padding
        self.inference_accelerate = inference_accelerate
        self.force_circular = force_circular

        self.sre_k = (kernel_size[0] // 2) + 1
        if not self.force_circular:
            self.sre_k += 1
        self.sre_k = self.sre_k if sre_k == None else sre_k

        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == "same":
                for d, k, i in zip(
                    dilation, kernel_size, range(len(kernel_size) - 1, -1, -1)
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )

        if transposed:
            self.weight = nn.Parameter(
                torch.empty(
                    (in_channels, out_channels // groups, self.sre_k), **factory_kwargs
                )
            )
            self.weight_matrix_shape = (
                in_channels,
                out_channels // groups,
                *kernel_size,
            )
            weight_index_mat = self._make_weight_index_mat(1, 1, factory_kwargs)
        else:
            self.weight = nn.Parameter(
                torch.empty(
                    (out_channels, in_channels // groups, self.sre_k), **factory_kwargs
                )
            )
            self.weight_matrix_shape = (
                out_channels,
                in_channels // groups,
                *kernel_size,
            )
            weight_index_mat = self._make_weight_index_mat(1, 1, factory_kwargs)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        # Fixed weight index matrix
        self.register_buffer("weight_index_mat", weight_index_mat)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        # manually init weight to avoid problem
        fan, _ = init._calculate_fan_in_and_fan_out(
            torch.zeros(self.weight_matrix_shape)
        )
        gain = init.calculate_gain("leaky_relu", math.sqrt(5))
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += ", padding_mode={padding_mode}"
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_SRE_ConvNd, self).__setstate__(state)
        if not hasattr(self, "padding_mode"):
            self.padding_mode = "zeros"

    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if self.inference_accelerate and not mode:
            self.infer_weight_matrix = self._make_conv_weight(self.weight)
        else:
            self.infer_weight_matrix = None
        return super().train(mode)

    def _make_conv_weight(self, weight):
        # Note that einsum is generally faster than batch matrix multiplication
        weight_index_mat = self.weight_index_mat

        weight = torch.einsum("ijkw,ijk->ijw", weight_index_mat, weight)
        weight = weight.reshape(self.weight_matrix_shape)
        return weight

    def _make_weight_index_mat(
        self, index_mat_C_in, index_mat_C_out, factory_kwargs
    ): ...

    def _conv_forward(
        self, input: Tensor, weight: Tensor, bias: Optional[Tensor]
    ) -> Tensor: ...


class SRE_Conv1d(_SRE_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        transposed: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        sre_k: int = None,
        inference_accelerate: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        dilation = _single(dilation)
        padding = padding if isinstance(padding, str) else _single(padding)
        super(SRE_Conv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            groups,
            bias,
            padding_mode,
            sre_k,
            inference_accelerate,
            **factory_kwargs
        )

    def _make_weight_index_mat(self, index_mat_C_in, index_mat_C_out, factory_kwargs):
        weight_index_mats = []
        _, _, H = self.weight_matrix_shape
        D = np.ones((H))
        D[H // 2] = 0
        D = ndimage.distance_transform_edt(D)
        max_dist = (H // 2) + 0.5 if self.force_circular else D.max()
        num_levels = self.sre_k + 1 if self.force_circular else self.sre_k
        levels = np.linspace(D.min(), max_dist, num=num_levels)
        for i in range(num_levels):
            if i == num_levels - 1:
                if self.force_circular:
                    continue
                idx = (D==levels[i]).astype(int)
            else:
                idx = ((D >= levels[i]) & (D < levels[i + 1])).astype(int)
            level_mat = torch.tensor(idx, **factory_kwargs)[None, None, :, :]
            weight_index_mats.append(level_mat.reshape(-1))
        weight_index_mats = torch.stack(weight_index_mats, dim=0)
        weight_index_mats = weight_index_mats.to(torch.float32)
        weight_index_mats = weight_index_mats.expand(
            (index_mat_C_in, index_mat_C_out, -1, -1)
        )
        return weight_index_mats

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return F.conv1d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _single(0),
                self.dilation,
                self.groups,
            )
        return F.conv1d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input):
        if not self.training and self.inference_accelerate:
            weight_matrix = self.infer_weight_matrix
            weight_matrix = weight_matrix.to(device=input.device, dtype=input.dtype)
        else:
            weight_matrix = self._make_conv_weight(self.weight)
        return self._conv_forward(input, weight_matrix, self.bias)


class SRE_Conv2d(_SRE_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        transposed: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        sre_k: int = None,
        inference_accelerate: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        dilation = _pair(dilation)
        padding = padding if isinstance(padding, str) else _pair(padding)
        super(SRE_Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            groups,
            bias,
            padding_mode,
            sre_k,
            inference_accelerate,
            **factory_kwargs
        )

    def _make_weight_index_mat(self, index_mat_C_in, index_mat_C_out, factory_kwargs):
        weight_index_mats = []
        _, _, H, W = self.weight_matrix_shape
        D = np.ones((H, W))
        D[(H // 2, W // 2)] = 0
        D = ndimage.distance_transform_edt(D)
        max_dist = (H // 2) + 0.5 if self.force_circular else D.max()
        num_levels = self.sre_k + 1 if self.force_circular else self.sre_k
        levels = np.linspace(D.min(), max_dist, num=num_levels)
        for i in range(num_levels):
            if i == num_levels - 1:
                if self.force_circular:
                    continue
                idx = (D==levels[i]).astype(int)
            else:
                idx = ((D >= levels[i]) & (D < levels[i + 1])).astype(int)
            level_mat = torch.tensor(idx, **factory_kwargs)[None, None, :, :]
            weight_index_mats.append(level_mat.reshape(-1))
        weight_index_mats = torch.stack(weight_index_mats, dim=0)
        weight_index_mats = weight_index_mats.to(torch.float32)
        weight_index_mats = weight_index_mats.expand(
            (index_mat_C_in, index_mat_C_out, -1, -1)
        )
        return weight_index_mats

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return F.conv2d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input):
        if not self.training and self.inference_accelerate:
            weight_matrix = self.infer_weight_matrix
            weight_matrix = weight_matrix.to(device=input.device, dtype=input.dtype)
        else:
            weight_matrix = self._make_conv_weight(self.weight)
        return self._conv_forward(input, weight_matrix, self.bias)


class SRE_Conv3d(_SRE_ConvNd):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        transposed: bool = False,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        sre_k: int = None,
        inference_accelerate: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        dilation = _triple(dilation)
        padding = padding if isinstance(padding, str) else _triple(padding)
        super(SRE_Conv3d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            transposed,
            groups,
            bias,
            padding_mode,
            sre_k,
            inference_accelerate,
            **factory_kwargs
        )

    def _make_weight_index_mat(self, index_mat_C_in, index_mat_C_out, factory_kwargs):
        weight_index_mats = []
        _, _, H, W, L = self.weight_matrix_shape
        D = np.ones((H, W, L))
        D[(H // 2, W // 2, L // 2)] = 0
        D = ndimage.distance_transform_edt(D)
        max_dist = (H // 2) + 0.5 if self.force_circular else D.max()
        num_levels = self.sre_k + 1 if self.force_circular else self.sre_k
        levels = np.linspace(D.min(), max_dist, num=num_levels)
        for i in range(num_levels):
            if i == num_levels - 1:
                if self.force_circular:
                    continue
                idx = (D==levels[i]).astype(int)
            else:
                idx = ((D >= levels[i]) & (D < levels[i + 1])).astype(int)
            level_mat = torch.tensor(idx, **factory_kwargs)[None, None, :, :, :]
            weight_index_mats.append(level_mat.reshape(-1))
        weight_index_mats = torch.stack(weight_index_mats, dim=0)
        weight_index_mats = weight_index_mats.to(torch.float32)
        weight_index_mats = weight_index_mats.expand(
            (index_mat_C_in, index_mat_C_out, -1, -1)
        )
        return weight_index_mats

    def _conv_forward(self, input, weight, bias):
        if self.padding_mode != "zeros":
            return F.conv3d(
                F.pad(
                    input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return F.conv3d(
            input, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, input):
        if not self.training and self.inference_accelerate:
            weight_matrix = self.infer_weight_matrix
            weight_matrix = weight_matrix.to(device=input.device, dtype=input.dtype)
        else:
            weight_matrix = self._make_conv_weight(self.weight)
        return self._conv_forward(input, weight_matrix, self.bias)


class SRE_ConvTranspose2d(SRE_Conv2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        sre_k=None,
        inference_accelerate: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        if padding_mode != "zeros":
            raise ValueError(
                'Only "zeros" padding mode is supported for {}'.format(
                    self.__class__.__name__
                )
            )
        super(SRE_ConvTranspose2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            sre_k,
            inference_accelerate,
            device,
            dtype,
        )

    # dilation being an optional parameter is for backwards
    # compatibility
    def _output_padding(
        self,
        input,
        output_size,
        stride,
        padding,
        kernel_size,
        num_spatial_dims,
        dilation=None,
    ):
        if output_size is None:
            ret = _single(self.output_padding)  # converting to list if was not already
        else:
            has_batch_dim = input.dim() == num_spatial_dims + 2
            num_non_spatial_dims = 2 if has_batch_dim else 1
            if len(output_size) == num_non_spatial_dims + num_spatial_dims:
                output_size = output_size[num_non_spatial_dims:]
            if len(output_size) != num_spatial_dims:
                raise ValueError(
                    "ConvTranspose{}D: for {}D input, output_size must have {} or {} elements (got {})".format(
                        num_spatial_dims,
                        input.dim(),
                        num_spatial_dims,
                        num_non_spatial_dims + num_spatial_dims,
                        len(output_size),
                    )
                )

            min_sizes = torch.jit.annotate(List[int], [])
            max_sizes = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                dim_size = (
                    (input.size(d + num_non_spatial_dims) - 1) * stride[d]
                    - 2 * padding[d]
                    + (dilation[d] if dilation is not None else 1)
                    * (kernel_size[d] - 1)
                    + 1
                )
                min_sizes.append(dim_size)
                max_sizes.append(min_sizes[d] + stride[d] - 1)

            for i in range(len(output_size)):
                size = output_size[i]
                min_size = min_sizes[i]
                max_size = max_sizes[i]
                if size < min_size or size > max_size:
                    raise ValueError(
                        (
                            "requested an output size of {}, but valid sizes range "
                            "from {} to {} (for an input of {})"
                        ).format(output_size, min_sizes, max_sizes, input.size()[2:])
                    )

            res = torch.jit.annotate(List[int], [])
            for d in range(num_spatial_dims):
                res.append(output_size[d] - min_sizes[d])

            ret = res
        return ret

    def forward(self, input, output_size=None):
        if self.padding_mode != "zeros":
            raise ValueError(
                "Only `zeros` padding mode is supported for ConvTranspose2d"
            )

        assert isinstance(self.padding, tuple)
        # One cannot replace List by Tuple or Sequence in "_output_padding" because
        # TorchScript does not support `Sequence[T]` or `Tuple[T, ...]`.
        num_spatial_dims = 2
        output_padding = self._output_padding(
            input,
            output_size,
            self.stride,
            self.padding,
            self.kernel_size,  # type: ignore[arg-type]
            num_spatial_dims,
            self.dilation,
        )  # type: ignore[arg-type]

        weight_matrix = self._make_conv_weight(self.weight)

        return F.conv_transpose2d(
            input,
            weight_matrix,
            self.bias,
            self.stride,
            self.padding,
            output_padding,
            self.groups,
            self.dilation,
        )
