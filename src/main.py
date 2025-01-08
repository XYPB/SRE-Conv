from __future__ import print_function
import os
from glob import glob
import shutil
import time
from pytz import timezone
from datetime import datetime
import json
import copy

import numpy as np
import torch
from torch.backends import cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LambdaLR, StepLR, MultiStepLR, CosineAnnealingLR
import torchvision.models.resnet as resnet
import medmnist
from medmnist import INFO
from sklearn.metrics import (roc_auc_score, accuracy_score, 
                             balanced_accuracy_score, confusion_matrix, 
                             f1_score)
import gc

from opt import get_opt
from SRE_Conv.transforms import get_medmnist_transforms, get_medmnist3d_transforms
import SRE_Conv.sre_resnet as sre_resnet
import SRE_Conv.sre_resnet_3d as sre_resnet_3d
from SRE_Conv.sre_conv import SRE_Conv2d
from utils import log_img, manual_seed, one_hot, log_train_val, acc_at_topk, force_cudnn_initialization

# @profile
def train(args, model, device, train_loader, optimizer, scaler, epoch, log_dir):
    if log_dir:
        train_log_dir = os.path.join(log_dir, 'vis_train')
        os.makedirs(train_log_dir, exist_ok=True)
    model.train()
    step_losses = []
    outputs = []
    targets = []
    step_grad_norm = []
    st = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if len(target.shape) == 2:
            target = target.squeeze(1)
        if args.dev:
            print(data.shape, target.shape)
            print(torch.unique(target.detach().cpu()))
        dtype = torch.bfloat16 if args.bf16 else None
        with torch.autocast('cuda', enabled=args.amp, dtype=dtype):
            output = model(data)
            if args.multi_label:
                loss = F.binary_cross_entropy_with_logits(output, target.to(torch.float32))
            else:
                loss = F.cross_entropy(output, target)
            if args.dev:
                print(data.shape, output.shape, data[:, 0, :10, 0], output)
        # autocast should only wrap the forward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        # cast torch bf16 to f32 for prediction output
        if args.bf16:
            output = output.detach().float()
        else:
            output = output.detach()

        # logging
        step_losses.append(loss.item())
        outputs.append(output.cpu().numpy())
        targets.append(target.detach().cpu().numpy())
        if args.log_grad_norm:
            grad_norm = 0
            for p in model.parameters():
                para_norm = p.grad.data.norm(2)
                grad_norm += para_norm.item() ** 2
            step_grad_norm.append(np.sqrt(grad_norm))
        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAvg. Loss: {np.mean(step_losses):.6f}\tTime: {time.time()-st:.2f}')
            if log_dir and args.log_img and epoch == 1:
                if data.shape[1] == 3:
                    img = data[0, :].detach().cpu().numpy()
                else:
                    img = data[0, 0].detach().cpu().numpy()
                img_label = target[0].detach().cpu().numpy()
                log_img(epoch, batch_idx, img, img_label, train_log_dir)
            if args.dry_run:
                return step_losses, 0, step_grad_norm
    et = time.time()
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0).squeeze()
    # Simply throw away the dummy class
    if args.multi_label:
        preds = (outputs > 0.5).astype(int)
        acc_top5 = 0.0
        auc = 100 * roc_auc_score(targets, outputs, multi_class='ovr')
        f1 = 100 * f1_score(targets, preds, average='macro')
        acc = 100 * accuracy_score(targets.flatten(), preds.flatten())
    else:
        preds = np.argmax(outputs, axis=1).squeeze()
        acc_top5 = 100 * acc_at_topk(targets, outputs, 5)
        auc = 100 * roc_auc_score(one_hot(targets, num_classes=outputs.shape[1]), outputs, multi_class='ovr')
        f1 = 100 * f1_score(targets, preds, average='macro')
        acc = 100 * accuracy_score(targets, preds)
    print(f"Train set average Acc@1: {acc:.2f}%,\t Acc@5: {acc_top5:.2f}%,\tAUC: {auc:.2f}%,\tF1-score: {f1:.2f}%,\tTime: {et-st:.2f}")
    del targets, outputs
    return step_losses, acc, step_grad_norm



def test(args, model, device, test_loader, epoch, verbose=True, confusion_mat=False):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    st = time.time()
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            if len(target.shape) == 2:
                target = target.squeeze(1)
            dtype = torch.bfloat16 if args.bf16 else None
            with torch.autocast('cuda', enabled=args.amp, dtype=dtype):
                output = model(data.to(torch.float32))
                if args.multi_label:
                    loss = F.binary_cross_entropy_with_logits(
                        output, target.to(torch.float32), reduction='sum')
                else:
                    loss = F.cross_entropy(output, target, reduction='sum')
            test_loss += loss.item()  # sum up batch loss
            if args.bf16:
                output = output.detach().float()
            else:
                output = output.detach()
            if args.log_img and epoch == 1 and batch_idx % 10 == 0:
                os.makedirs('./tmp/vis_test', exist_ok=True)
                if data.shape[1] == 3:
                    img = data[0, :].detach().cpu().numpy()
                else:
                    img = data[0, 0].detach().cpu().numpy()
                img_label = target[0].detach().cpu().numpy()
                log_img(0, batch_idx, img, img_label, './tmp/vis_test')
            outputs.append(output.cpu().numpy())
            targets.append(target.detach().cpu().numpy())
    et = time.time()

    test_loss /= len(test_loader.dataset)
    outputs = np.concatenate(outputs, axis=0)
    targets = np.concatenate(targets, axis=0).squeeze()
    # Simply throw away the dummy class
    if args.multi_label:
        preds = (outputs > 0.5).astype(int)
        acc_top5 = 0.0
        auc = 100 * roc_auc_score(targets, outputs, multi_class='ovr')
        f1 = 100 * f1_score(targets, preds, average='macro')
        ba = 0.0
        acc = 100 * accuracy_score(targets.flatten(), preds.flatten())
    else:
        preds = np.argmax(outputs, axis=1).squeeze()
        acc_top5 = 100 * acc_at_topk(targets, outputs, 5)
        auc = 100 * roc_auc_score(one_hot(targets, num_classes=outputs.shape[1]), outputs, multi_class='ovr')
        f1 = 100 * f1_score(targets, preds, average='macro')
        ba = 100 * balanced_accuracy_score(targets, preds)
        acc = 100 * accuracy_score(targets, preds)
    correct = np.sum(targets == preds)

    if verbose:
        print(f'\nTest set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({acc:.2f}%), Acc@5: {acc_top5:.2f}%, AUC: {auc:.2f}%, BA: {ba:.2f}%, F1-score: {f1:.2f}% Time: {et-st:.2f}\n')
        if confusion_mat:
            print(f'Confusion Matrix:\n{confusion_matrix(targets, preds)}\n')
    return test_loss, acc


def eval_rot(model, test_loader, device, args, verbose=False, eval_3d=False):
    overall_acc = []
    print('## Evaluate on rotation:')
    degrees = range(0, 361, 30) if eval_3d else range(0, 361, 10)
    axes = ['x', 'y', 'z'] if eval_3d else [''] 
    for ax in axes:
        for deg in degrees:
            if verbose:
                print(f'## Evaluate on rotation: {deg} and axis: {ax}')
            args.degree = deg
            args.fix_rotate = True
            if eval_3d:
                args.aug_axis = ax
                _, test_transform = get_medmnist3d_transforms(args)
            else:
                _, test_transform = get_medmnist_transforms(args)
            # hack to test rotation
            test_loader.dataset.transform = test_transform
            test_loss, test_acc = test(args, model, device, test_loader, epoch=0, verbose=verbose)
            overall_acc.append(round(test_acc, 2))
    print(f'## Overall Acc. for all degrees: {overall_acc}')
    print(f'## Overall Avg. Acc.: {np.round(np.mean(overall_acc), 2)}({np.round(np.std(overall_acc), 2)})')
    return np.mean(overall_acc)


def eval_flip(model, test_loader, device, args, verbose=False, eval_3d=False):
    overall_acc = []
    flipping = ['', 'v', 'h', 'd'] if eval_3d else ['', 'v', 'h']
    print('## Evaluate on flipping:')
    for flip in flipping:
        if verbose:
            print(f'## Evaluate on flip: {flip}')
        if 'v' in flip:
            args.vflip = True
        if 'h' in flip:
            args.hflip = True
        if 'd' in flip:
            args.dflip = True
        if eval_3d:
            _, test_transform = get_medmnist3d_transforms(args)
        else:
            _, test_transform = get_medmnist_transforms(args)
        # hack to test rotation
        test_loader.dataset.transform = test_transform
        test_loss, test_acc = test(args, model, device, test_loader, epoch=0, verbose=verbose)
        overall_acc.append(round(test_acc, 2))
        # Reset
        args.vflip = False
        args.hflip = False
        args.dflip = False
    print(f'## Overall Acc. for all flipping: {overall_acc}')
    print(f'## Overall Avg. Acc.: {np.round(np.mean(overall_acc[1:]), 2)}({np.round(np.std(overall_acc), 2)})')
    return np.mean(overall_acc)


# @profile
def main_worker(args):
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    manual_seed(args.seed)
    if args.save_model or args.log:
        est = timezone('US/Eastern')
        dt = est.localize(datetime.now())
        dt_str = dt.strftime('%Y-%m-%d-%H-%M-%S')
        task_name = args.med_mnist
        log_dir = os.path.join(args.base_log_dir, f'{task_name}_{dt_str}_{args.model_type}_{args.exp}_train_logs')
        config_dir = os.path.join(log_dir, 'config')
    else:
        log_dir = None

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.cudnn:
        cudnn.benchmark = True

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.batch_size}
    if use_cuda:
        num_cores = len(os.sched_getaffinity(0))
        if args.num_workers > num_cores:
            args.num_workers = num_cores
        print(f'### Use {args.num_workers} cores for training...')
        cuda_kwargs = {'num_workers': args.num_workers,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)
    if 'ric' in args.model_type:
        train_kwargs['drop_last'] = True
        test_kwargs['drop_last'] = True

    ####### DATASET #######
    in_channels = 3
    info = INFO[args.med_mnist]
    in_channels = info['n_channels']
    n_classes = len(info['label'])
    if '3d' in args.med_mnist:
        transform, test_transform = get_medmnist3d_transforms(args)
        input_shape = (in_channels, 32, 32, 32)
    else:
        transform, test_transform = get_medmnist_transforms(args)
        input_shape = (in_channels, 32, 32)
    DataClass = getattr(medmnist, info['python_class'])
    train_dataset = DataClass(split='train', transform=transform, download=True, 
                                size=args.medmnist_size)
    test_dataset = DataClass(split='test', transform=test_transform, download=True,
                                size=args.medmnist_size)

    ####### MODEL #######

    if '3d' in args.med_mnist:
        if args.sre_conv_size_list != None:
            sre_conv_size = args.sre_conv_size_list
        else:
            sre_conv_size = args.sre_conv_size
        model = getattr(sre_resnet_3d, args.model_type)(num_classes=n_classes,
                                                       in_channels=in_channels,
                                                       sre_conv_size=sre_conv_size,
                                                       sre_k=args.sre_conv_k,)
    elif "sre" in args.model_type:
        if args.sre_conv_size_list != None:
            sre_conv_size = args.sre_conv_size_list
        else:
            sre_conv_size = args.sre_conv_size
        model = getattr(sre_resnet, args.model_type)(num_classes=n_classes, 
                                                    inplanes=args.res_inplanes,
                                                    sre_conv_size=sre_conv_size, 
                                                    large_conv=args.large_conv,
                                                    sre_k=args.sre_conv_k,
                                                    in_channels=in_channels)
        # remove the first downsampling to ensure last stage input size
        if args.res_keep_conv1:
            model.conv1 = SRE_Conv2d(in_channels, args.res_inplanes, kernel_size=5, stride=1, 
                                    padding=2, bias=False)
        model.maxpool = nn.Identity()
    else:
        model = getattr(resnet, args.model_type)(num_classes=n_classes)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    if args.maxpool:
        model.avgpool = nn.AdaptiveMaxPool2d((1, 1))

    if args.dev:
        print(model)

    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)
    model = model.to(device)

    #### Scaler
    scaler = GradScaler(enabled=args.scaler)

    param_cnt = sum([param.numel() for param in model.parameters() if param.requires_grad])
    print(f'Correct Total params: {param_cnt}')

    param_group = [{'params': model.parameters()}]
    if args.adam:
        optimizer = optim.Adam(param_group, lr=args.lr, weight_decay=args.weight_decay)
    elif args.sgd:
        optimizer = optim.SGD(param_group, lr=args.lr, momentum=args.momentum, 
                              weight_decay=args.weight_decay)
    elif args.adamW:
        optimizer = optim.AdamW(param_group, lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.Adadelta(param_group, lr=args.lr)

    torch.autograd.set_detect_anomaly(True)

    cur_ep = 1
    if args.resume:
        args.log = True
        args.save_model = True
        print(f'### resume experiment logged under {args.resume}...')
        log_dir = args.resume
        config_dir = os.path.join(log_dir, 'config')
        ckpt_dir = os.path.join(log_dir, 'ckpt')
        assert os.path.exists(log_dir)
        assert os.path.exists(config_dir)
        assert os.path.exists(ckpt_dir)
        args.load_model = log_dir
    elif args.log or args.save_model:
        os.makedirs(log_dir, exist_ok=False)
        print(f'### experiment logged under {log_dir}...')
        os.makedirs(config_dir, exist_ok=True)
        arg_dict = vars(args)
        json.dump(arg_dict, open(os.path.join(config_dir, 'train_config.json'), 'w'))

    if args.load_model:
        assert os.path.exists(args.load_model)
        print(f'### load model logged under {args.load_model}...')
        model_ckpt_dir = os.path.join(args.load_model, 'ckpt')
        ckpts = sorted(glob(os.path.join(model_ckpt_dir, '*.ckpt')))
        ckpt_path = ckpts[-1]
        ckpt = torch.load(ckpt_path)
        state_dict = ckpt['state_dict']
        cur_ep = ckpt['cur_ep']
        print(f'### load model {ckpt_path} at epoch {cur_ep}...')
        target_state_dict = {}
        for k, param in state_dict.items():
            k = k.replace('module.', '')
            target_state_dict[k] = param
        model.load_state_dict(target_state_dict)

    if cur_ep-2 != -1:
        # init initial_lr
        for group in optimizer.param_groups:
            group.setdefault('initial_lr', args.lr)
    if args.cos:
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.min_lr, last_epoch=cur_ep-2)
    elif args.step:
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma, last_epoch=cur_ep-2)
    elif args.multi_step:
        scheduler = MultiStepLR(optimizer, [50, 75], gamma=0.1, last_epoch=cur_ep-2)
    else:
        scheduler = LambdaLR(optimizer, lambda x: x, last_epoch=cur_ep-2)

    total_train_loss = []
    total_train_acc = []
    total_grad_norm = []
    total_test_loss = []
    total_test_acc = []
    total_eval_rot_acc = []
    total_eval_flip_acc = []
    best_acc = -1
    for epoch in range(cur_ep, args.epochs + 1):

        step_losses, train_acc, grad_norm = train(
            args, model, device, train_loader, optimizer, scaler, epoch, log_dir, 
        )
        test_loss, test_acc = test(args, model, device, test_loader, epoch=epoch, confusion_mat=False)
        steps = len(step_losses)
        total_train_loss += step_losses
        total_train_acc += [train_acc for _ in range(steps)]
        if args.log_grad_norm:
            total_grad_norm += grad_norm
        total_test_loss += [test_loss for _ in range(steps)]
        total_test_acc += [test_acc for _ in range(steps)]
        scheduler.step()
        if epoch == 1 and args.dev:
            print(torch.cuda.memory_summary(device=device, abbreviated=False))

        if args.log:
            total_grad_norm = total_grad_norm if args.log_grad_norm else None
            log_train_val(total_train_loss, total_test_loss, 
                        total_train_acc, total_test_acc, total_grad_norm, log_dir)

        if args.save_model:
            ckpt_dir = os.path.join(log_dir, 'ckpt')
            os.makedirs(ckpt_dir, exist_ok=True)
            if args.save_rec and ((epoch + 1) % args.save_interval) == 0:
                ckpt = {
                    "state_dict": model.state_dict(),
                    "cur_ep": epoch + 1,
                }
                torch.save(ckpt, os.path.join(ckpt_dir, f"{args.model_type}_ep{epoch:0>4d}.ckpt"))
            elif args.save_best:
                ckpt_dest = os.path.join(ckpt_dir, f"{args.model_type}_last.ckpt")
                ckpt = {
                    "state_dict": model.state_dict(),
                    "cur_ep": epoch + 1,
                }
                torch.save(ckpt, ckpt_dest)
                if test_acc > best_acc:
                    best_dist = os.path.join(ckpt_dir, f"{args.model_type}_best.ckpt")
                    print(f'### Update best weight with test auc: {test_acc:.4f}')
                    shutil.copy(ckpt_dest, best_dist)
                    best_acc = test_acc
            else:
                ckpt_dest = os.path.join(ckpt_dir, f"{args.model_type}_last.ckpt")
                ckpt = {
                    "state_dict": model.state_dict(),
                    "cur_ep": epoch + 1,
                }
                torch.save(ckpt, ckpt_dest)

        gc.collect()

        verbose = (epoch == args.epochs)
        eval_3d = (args.med_mnist != None and '3d' in args.med_mnist)
        if args.eval_rot and (epoch % args.eval_interval == 0 or epoch == args.epochs):
            # don't change the original args
            rot_acc = eval_rot(model, test_loader, device, 
                               copy.deepcopy(args), verbose=verbose, eval_3d=eval_3d)
            total_eval_rot_acc.append(rot_acc)

        if args.eval_flip and (epoch % args.eval_interval == 0 or epoch == args.epochs):
            # don't change the original args
            flip_acc = eval_flip(model, test_loader, device, 
                                 copy.deepcopy(args), verbose=verbose, eval_3d=eval_3d)
            total_eval_flip_acc.append(flip_acc)
    
    if args.log:
        with open(os.path.join(log_dir, 'logs.json'), 'w') as f:
            serialize = lambda l: [float(x) for x in l]
            logs = {
                'train_loss': serialize(total_train_loss),
                'test_loss': serialize(total_test_loss),
                'train_acc': serialize(total_train_acc),
                'test_acc': serialize(total_test_acc),
                'eval_rot_acc': serialize(total_eval_rot_acc),
                'eval_flip_acc': serialize(total_eval_flip_acc),
            }
            json.dump(logs, f)


if __name__ == '__main__':
    args = get_opt()
    main_worker(args)