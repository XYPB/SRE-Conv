# SRE-Conv: Symmetric Rotation Equivariant Convolution for Biomedical Image Classification

[![PyPI version](https://img.shields.io/pypi/v/SRE-Conv.svg)](https://pypi.org/project/SRE-Conv/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![arXiv:2501.09753](https://img.shields.io/badge/arXiv-2501.09753-B31B1B.svg)](https://arxiv.org/abs/2501.09753)

This is the official implementation of paper "SRE-Conv: Symmetric Rotation Equivariant Convolution for Biomedical Image Classification" (accepted by **ISBI 2025**)

*[Yuexi Du](https://xypb.github.io/), Jiazhen Zhang, Tal Zeevi, [Nicha C. Dvornek](https://www.hellonicha.com/), [John A. Onofrey](https://medicine.yale.edu/profile/john-onofrey/)*

*Yale University*

![teaser](assets/tesser_480p.gif)

For any question related to the implementation, please open issue in this repo, we will respond to you ASAP. For other question about the paper, please contact the author directly.

## News

- **Jan. 2025** Our paper is now available on [arXiv](https://arxiv.org/abs/2501.09753)!
- **Jan. 2025** Paper accepted by ISBI 2025 and GitHub code released🎉!

## Abstract

> Convolutional neural networks (CNNs) are essential tools for computer vision tasks, but they lack traditionally desired properties of extracted features that could further improve model performance, e.g., rotational equivariance. Such properties are ubiquitous in biomedical images, which often lack explicit orientation. While current work largely relies on data augmentation or explicit modules to capture orientation information, this comes at the expense of increased training costs or ineffective approximations of the desired equivariance. To overcome these challenges, we propose a novel and efficient implementation of the Symmetric Rotation-Equivariant (SRE) Convolution (SRE-Conv) kernel, designed to learn rotation-invariant features while simultaneously compressing the model size. The SRE-Conv kernel can easily be incorporated into any CNN backbone. We validate the ability of a deep SRE-CNN to capture equivariance to rotation using the public MedMNISTv2 dataset (16 total tasks). SRE-Conv- CNN demonstrated improved rotated image classification performance accuracy on all 16 test datasets in both 2D and 3D images, all while increasing efficiency with fewer parameters and reduced memory footprint.


## Installation

We provide both the PyPI package for [SRE-Conv](https://pypi.org/project/SRE-Conv/) and the code to reproduce the experiment results in this repo.

To install and directly use the SRE-Conv, please run the following command:
```bash
pip install SRE-Conv
```

The minimal requirement for the SRE-Conv is:
```bash
"scipy>=1.9.0",
"numpy>=1.22.0",
"torch>=1.8.0"
```

**Note**: Using lower version of torch and numpy should be fine given that we didn't use any new feature in the new torch version, but we do suggest you to follow the required dependencies. If you have to use the different version of torch/numpy, you may also try to install the package from source code at [project repo](https://github.com/XYPB/SRE-Conv).

## Usage

Our SRE-Conv is implemented with the same interface as conventional torch convolutional layer. It can be used easily in any modern deep learning CNN implemented in PyTorch.

```python
import torch
from SRE_Conv import SRE_Conv2d

x = torch.randn(2, 3, 32, 32)
# create a 2D SRE-Conv of size 3x3
sre_conv = SRE_Conv2d(3, 16, 3)
y = SRE_conv(x)
x_rot = torch.rot90(x, 1, (2, 3))
y_rot = SRE_conv(x)
# check equivariance under 90-degree rotation
print(torch.allclose(torch.rot90(y, 1, (2, 3)), y_rot))
```

For more detail about the specific argument for our SRE-Conv, please refer to [here](https://github.com/XYPB/SRE-Conv/blob/458e24c61f97229cfa167c60ad03f7f2c43bb91e/src/SRE_Conv/sre_conv.py#L40-L71).

We have also provided SRE-ResNet and SRE-ResNet3D in this repo, you may also use it as regular ResNet but with rotational equivariance.

```python
import torch
from SRE_Conv import sre_resnet18

x = torch.randn(2, 3, 32, 32)
# use "sre_conv_size" argument to specify kernel size at each stage.
sre_r18 = sre_resnet18(sre_conv_size=[9, 9, 5, 5])
output = sre_r18(x)
```

For general CNN implemented in PyTorch, you may use ``convert_to_SRE_conv`` function to convert it from regular CNN to equivariant CNN using ``SRE_Conv2d``.

```python
import torch
import torchvision.models.resnet as resnet
from SRE_Conv import convert_to_SRE_conv

model = resnet.resnet18()
sre_model = convert_to_SRE_conv(model)
```


## Train & Evaluation on MedMNIST

To reproduce the experiment results, you may also need to install the following packages:
```bash
"medmnist>=3.0.0" # This includes all the datasets we used in the paper
"grad-cam>=1.5.0"
"matplotlib"
"imageio"
```

Run the following comment to train the model and evaluate the performance under both flip and rotation evaluation.
```bash
cd ./src
# 2D MedMNIST
python main.py --med-mnist <medmnist_dataset> --epochs 100 --model-type sre_resnet18 --sre-conv-size-list 9 9 5 5 -b 128 --lr 2e-2 --cos --sgd --eval-rot --eval-flip --train-flip-p 0 --log --cudnn --moco-aug --translate-ratio 0.1 --translation --save-model  --save-best --res-keep-conv1
# 3D MedMNIST
python main.py --med-mnist <medmnist3d_dataset> --epochs 100 --model-type sre_r3d_18 --ri-conv-size-list 5 5 5 5 -b 4 --lr 1e-2 --cos --sgd --eval-rot --res-keep-conv1 --log --cudnn --moco-aug
```


## Reference

```
@article{du2025sreconv,
      title={SRE-Conv: Symmetric Rotation Equivariant Convolution for Biomedical Image Classification}, 
      author={Du, Yuexi and Onofrey, John A and others},
      journal={arXiv preprint arXiv:2501.09753},
      year={2025},
      eprint={2501.09753},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2501.09753}, 
}
```
