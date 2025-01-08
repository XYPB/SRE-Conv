import torch
import torchio as tio
import torchvision.transforms.functional as F
from torchvision import transforms


class DummyToTensor(object):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return torch.tensor(x)


class PadRotateWrapper(object):
    def __init__(self, rotate_trans, padding="zeros", img_size=32):
        super().__init__()
        self.rotate_trans = rotate_trans
        self.padding_mode = padding
        if padding != "zeros":
            self.to_pad = int(img_size // 2)
        else:
            self.to_pad = None
        self.img_size = img_size

    def __call__(self, x):
        # pad the image before rotate
        if self.to_pad != None:
            x = F.pad(
                x,
                (self.to_pad, self.to_pad, self.to_pad, self.to_pad),
                padding_mode=self.padding_mode,
            )
        x = self.rotate_trans(x)
        # crop the image after rotate
        if self.to_pad != None:
            x = F.center_crop(x, [self.img_size, self.img_size])
        return x


class FixRotate(object):
    def __init__(self, degree, expand=False):
        super().__init__()
        self.degree = degree
        self.expand = expand

    def __call__(self, x):
        return F.rotate(x, self.degree, expand=self.expand)


class PadTransWrapper(object):
    def __init__(self, trans, padding="zeros", img_size=32):
        super().__init__()
        self.trans = trans
        self.padding_mode = padding
        if padding != "zeros":
            self.to_pad = int(img_size // 2)
        else:
            self.to_pad = None
        self.img_size = img_size

    def __call__(self, x):
        # pad the image before rotate
        if self.to_pad != None:
            x = F.pad(
                x,
                (self.to_pad, self.to_pad, self.to_pad, self.to_pad),
                padding_mode=self.padding_mode,
            )
        x = self.trans(x)
        # crop the image after rotate
        if self.to_pad != None:
            x = F.center_crop(x, [self.img_size, self.img_size])
        return x


class FixRotateAxis(object):
    def __init__(self, degree, axis="x", expand=False, interpolation="nearest"):
        super().__init__()
        self.axis = axis
        self.degree = degree
        self.expand = expand
        if interpolation == "nearest":
            self.interpolation = F.InterpolationMode.NEAREST
        elif interpolation == "bilinear":
            self.interpolation = F.InterpolationMode.BILINEAR
        elif interpolation == "bicubic":
            self.interpolation = F.InterpolationMode.BICUBIC

    def __call__(self, x):
        if self.axis == "x":
            x = F.rotate(
                x, self.degree, expand=self.expand, interpolation=self.interpolation
            )
        elif self.axis == "y":
            x = F.rotate(
                x.permute(0, 2, 1, 3),
                self.degree,
                expand=self.expand,
                interpolation=self.interpolation,
            )
            x = x.permute(0, 2, 1, 3)
        elif self.axis == "z":
            x = F.rotate(
                x.permute(0, 3, 1, 2),
                self.degree,
                expand=self.expand,
                interpolation=self.interpolation,
            )
            x = x.permute(0, 2, 3, 1)
        else:
            raise ValueError("axis should be x, y or z")
        return x


class PadTransWrapper3D(PadTransWrapper):
    def __init__(self, trans, padding="zeros", img_size=32):
        super().__init__(trans, padding, img_size)

    def __call__(self, x):
        if self.to_pad != None:
            x = F.pad(
                x,
                (
                    self.to_pad,
                    self.to_pad,
                    self.to_pad,
                    self.to_pad,
                    self.to_pad,
                    self.to_pad,
                ),
                padding_mode=self.padding_mode,
            )
        x = self.trans(x)
        if self.to_pad != None:
            # 3D center crop
            _, H, W, D = x.shape
            Hs = (H - self.img_size) // 2
            Ws = (W - self.img_size) // 2
            Ds = (D - self.img_size) // 2
            x = x[
                :,
                Hs : Hs + self.img_size,
                Ws : Ws + self.img_size,
                Ds : Ds + self.img_size,
            ]
        return x


def get_medmnist_transforms(args):
    target_size = 32 if args.medmnist_size == 28 else args.medmnist_size
    if args.moco_aug:
        transform = [
            transforms.Resize((target_size, target_size)),
            transforms.RandomResizedCrop(target_size, scale=(0.5, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.1, saturation=0.1)],
                p=0.5,  # not strengthened
            ),
            # transforms.RandomGrayscale(p=0.2),
            # transforms.RandomApply([transforms.GaussianBlur((3, 3), [0.1, 2.0])], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    else:
        transform = [
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]

    test_transform = [
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    if args.fix_rotate:
        test_transform.append(
            PadTransWrapper(
                FixRotate(
                    args.degree, expand=args.expand, interpolation=args.interpolation
                ),
                padding=args.padding,
                img_size=target_size,
            )
        )
    elif args.rotate or args.train_rotate:
        test_transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.degree, expand=args.expand),
                padding=args.padding,
                img_size=target_size,
            )
        )
    if args.train_rotate and args.fix_rotate:
        transform.append(
            PadTransWrapper(
                FixRotate(
                    args.degree, expand=args.expand, interpolation=args.interpolation
                ),
                padding=args.padding,
                img_size=target_size,
            )
        )
    elif args.train_rotate:
        transform.append(
            PadTransWrapper(
                transforms.RandomRotation(args.degree, expand=args.expand),
                padding=args.padding,
                img_size=target_size,
            )
        )
    if args.translation:
        transform.append(
            PadTransWrapper(
                transforms.RandomAffine(
                    0, (args.translate_ratio, args.translate_ratio)
                ),
                padding=args.padding,
                img_size=target_size,
            )
        )
    if args.test_translation:
        args.test_translate_ratio = (
            args.translate_ratio
            if args.test_translate_ratio is None
            else args.test_translate_ratio
        )
        test_transform.append(
            PadTransWrapper(
                transforms.RandomAffine(
                    0, (args.test_translate_ratio, args.test_translate_ratio)
                ),
                padding=args.padding,
                img_size=target_size,
            )
        )
    if args.vflip:
        test_transform.append(transforms.RandomVerticalFlip(p=1.0))
    if args.hflip:
        test_transform.append(transforms.RandomHorizontalFlip(p=1.0))
    transform = transforms.Compose(transform)
    test_transform = transforms.Compose(test_transform)
    return transform, test_transform


def get_medmnist3d_transforms(args):
    target_size = 32 if args.medmnist_size == 28 else args.medmnist_size
    if args.moco_aug:
        transform = [
            DummyToTensor(),
            tio.transforms.Resize((target_size, target_size, target_size)),
            tio.transforms.RandomAffine(scales=0.2, translation=3, degrees=0),
            tio.transforms.RandomBlur(0.5),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    else:
        transform = [
            DummyToTensor(),
            tio.transforms.Resize((target_size, target_size, target_size)),
            transforms.Normalize((0.5,), (0.5,)),
        ]

    test_transform = [
        DummyToTensor(),
        tio.transforms.Resize((target_size, target_size, target_size)),
        transforms.Normalize((0.5,), (0.5,)),
    ]
    if args.fix_rotate:
        test_transform.append(
            PadTransWrapper3D(
                FixRotateAxis(
                    args.degree,
                    expand=args.expand,
                    axis=args.aug_axis,
                    interpolation=args.interpolation,
                ),
                padding=args.padding,
                img_size=target_size,
            )
        )
    elif args.rotate or args.train_rotate:
        test_transform.append(
            PadTransWrapper3D(
                tio.transforms.RandomAffine(scales=0, degress=args.degree),
                padding=args.padding,
                img_size=target_size,
            )
        )
    if args.train_rotate and args.fix_rotate:
        transform.append(
            PadTransWrapper3D(
                FixRotateAxis(
                    args.degree,
                    expand=args.expand,
                    axis=args.aug_axis,
                    interpolation=args.interpolation,
                ),
                padding=args.padding,
                img_size=target_size,
            )
        )
    elif args.train_rotate:
        transform.append(
            PadTransWrapper3D(
                tio.transforms.RandomAffine(scales=0, degress=args.degree),
                padding=args.padding,
                img_size=target_size,
            )
        )

    if args.vflip:
        test_transform.append(tio.transforms.RandomFlip(axes=1, flip_probability=1.0))
    if args.hflip:
        test_transform.append(tio.transforms.RandomFlip(axes=2, flip_probability=1.0))
    if args.dflip:
        test_transform.append(tio.transforms.RandomFlip(axes=0, flip_probability=1.0))
    transform = transforms.Compose(transform)
    test_transform = transforms.Compose(test_transform)
    return transform, test_transform
