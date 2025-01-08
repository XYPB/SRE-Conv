# "SRE-Conv: Symmetric Rotation Equivariant Convolution for Biomedical Image Classification"
*Yuexi Du, Jiazhen Zhang, Tal Zeevi, Nicha C. Dvornek, John A. Onofrey*

*Yale University*

![teaser](assets/tesser_480p.gif)

This is the official implementation of paper "SRE-Conv: Symmetric Rotation Equivariant Convolution for Biomedical Image Classification" (accepted by ISBI 2025)

### Abstract

> Convolutional neural networks (CNNs) are essential tools for computer vision tasks, but they lack traditionally desired properties of extracted features that could further improve model performance, e.g., rotational equivariance. Such properties are ubiquitous in biomedical images, which often lack explicit orientation. While current work largely relies on data augmentation or explicit modules to capture orientation information, this comes at the expense of increased training costs or ineffective approximations of the desired equivariance. To overcome these challenges, we propose a novel and efficient implementation of the Symmetric Rotation-Equivariant (SRE) Convolution (SRE-Conv) kernel, designed to learn rotation-invariant features while simultaneously compressing the model size. The SRE-Conv kernel can easily be incorporated into any CNN backbone. We validate the ability of a deep SRE-CNN to capture equivariance to rotation using the public MedMNISTv2 dataset (16 total tasks). SRE-Conv- CNN demonstrated improved rotated image classification performance accuracy on all 16 test datasets in both 2D and 3D images, all while increasing efficiency with fewer parameters and reduced memory footprint.


### News

- **Jan. 2025** Paper accepted by ISBI 2025 and GitHub code released

### Installation

We provide both the PyPI package for SRE-Conv and the code to reproduce the experiment results in this repo

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

### Usage
```python
>>> import torch
>>> from SRE_Conv import SRE_Conv2d, sre_resnet18
>>> x = torch.randn(2, 3, 32, 32)
>>> SRE_conv = SRE_Conv2d(3, 16, 3)
>>> conv_out = SRE_conv(x)
>>> SRE_r18 = sre_resnet18()
>>> output = SRE_r18(x)
# To reproduce the SRE-ResNet18 used in the paper, use:
>>> SRE_r18 = sre_resnet18(sre_conv_size=[9, 9, 5, 5], skip_first_maxpool=True)
```

### Train & Evaluation on MedMNIST

To reproduce the experiment results, you may also need to install the following packages:
```bash
"medmnist>=3.0.0"
"grad-cam>=1.5.0"
"matplotlib"
"imageio"
```

Run the following comment to train the model and evaluate the performance under both flip and rotation evaluation.
```bash
python main.py --med-mnist <medmnist_dataset> --epochs 100 --model-type sre_resnet18 --sre-conv-size-list 9 9 5 5 -b 128 --lr 2e-2 --cos --sgd --eval-rot --eval-flip --train-flip-p 0 --log --cudnn --moco-aug --translate-ratio 0.1 --translation --save-model --save-best
```


### Reference

*We will update this soon...*