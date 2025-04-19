import argparse
import torch
from copy import deepcopy
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import time
import os
import sys
import numpy as np

from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagnet_prompts import imagenet_classes
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed, setup_logger
from clip.custom_clip_batch_single import get_coop, get_coop2
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset, WeakStrongAugmenter
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


def select_confident_samples_pred(logits, thred=0.7):
    _, preds = logits.detach().topk(1, 1, True, True)
    preds = preds.squeeze(1)
    idx = []
    logits_softmax = logits.detach().clone().softmax(1)
    max_idx, max_val = -1, -1
    for i, p in enumerate(preds):
        if logits_softmax[i][p] > thred:
            idx.append(i)
        if logits_softmax[i][p] > max_val:
            max_idx = i
            max_val = logits_softmax[i][p]
    idx = torch.tensor(idx).cuda()
    if len(idx) == 0:
        idx = torch.tensor([max_idx]).cuda()

    return logits[idx], idx


def select_confident_samples_ratio(logits, ratio=0.1):
    logits_softmax = logits.detach().clone().softmax(1)
    val, preds = logits_softmax.detach().topk(1, 1, True, True)
    idx = torch.argsort(-val.squeeze(1), dim=0)[:int(len(val)*ratio)]
    return logits[idx], idx


def dist_loss_l1(output_teacher, student_output, temp=1.0):
    return torch.abs(output_teacher-student_output).sum(dim=1).mean()


def dist_loss_l2(output_teacher, student_output, temp=1.0):
    return torch.sqrt(((output_teacher-student_output)**2).sum(dim=1)).mean()


def div(logits, epsilon=1e-8):
    probs = F.softmax(logits, dim=1)
    probs_mean = probs.mean(dim=0)
    loss_div = -torch.sum(-probs_mean * torch.log(probs_mean + epsilon))
    return loss_div


def main_worker(args):
    # Create CLIP model
    model = get_coop2(args.arch, args.dataset, args.gpu, args.n_ctx, args.ctx_init)
    # Only tuning the prompt parameters
    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)
    print("=> Model created: visual backbone {}".format(args.arch))

    torch.cuda.set_device(args.gpu)
    model = model.cuda(args.gpu)

    # Define optimizer
    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.s1_lr)
    optim_state = deepcopy(optimizer.state_dict())
    # Setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # Norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # Iterating through eval datasets
    results = {}

    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    data_transform = WeakStrongAugmenter(base_transform, preprocess)

    set_id = args.dataset
    print("evaluating: {}".format(set_id))

    # Reset the model
    # Reset classnames of custom CLIP model
    if len(set_id) > 1:
        # Fine-grained classification datasets
        classnames = eval("{}_classes".format(set_id.lower()))
    else:
        assert set_id in ['A', 'R', 'K', 'V', 'I']
        classnames_all = imagenet_classes
        classnames = []
        if set_id in ['A', 'R', 'V']:
            label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
            if set_id == 'R':
                for i, m in enumerate(label_mask):
                    if m:
                        classnames.append(classnames_all[i])
            else:
                classnames = [classnames_all[i] for i in label_mask]
        else:
            classnames = classnames_all

    model.reset_classnames(classnames, args.arch)

    val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
    print("number of test samples: {}".format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.s1_batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    # ------ Stage I ------
    for epoch in range(args.s1_epoch):
        print("**Stage I Train epoch [{}/{}], lr={}:".format(epoch+1, args.s1_epoch, optimizer.param_groups[0]['lr']))
        results[set_id] = train_stage1_one_epoch(val_loader, model, optimizer, optim_state, scaler, args)
        print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        print("**Stage I Test after epoch [{}/{}]:".format(epoch+1, args.s1_epoch))
        test(val_loader, model, args)

        lr = args.s1_lr / 10**(epoch+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        torch.save({"ctx": model.prompt_learner.ctx}, os.path.join(args.output_dir, "ctx_"+str(epoch)+".pth"))
    
    # ------ Stage II ------
    with torch.no_grad():
        model.prompt_learner.ctx_init_state = model.prompt_learner.ctx.detach().clone()

    for name, param in model.named_parameters():
        if "prompt_learner" not in name:
            param.requires_grad_(False)

    trainable_param = model.prompt_learner.parameters()
    optimizer = torch.optim.AdamW(trainable_param, args.s2_lr)
    optim_state = deepcopy(optimizer.state_dict())

    base_transform = transforms.Compose([
        transforms.Resize(args.resolution, interpolation=BICUBIC),
        transforms.CenterCrop(args.resolution)])
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize])
    data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.s2_views - 1, augmix=len(set_id) > 1)

    val_dataset = build_dataset(set_id, data_transform, args.data, mode=args.dataset_mode)
    print("number of test samples: {}".format(len(val_dataset)))
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler, args)


def train_stage1_one_epoch(val_loader, model, optimizer, optim_state, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Train & Test: ')

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        for k in range(len(images)):
            images[k] = images[k].cuda(args.gpu, non_blocking=True)
        images = torch.cat(images, dim=0)
        ori_images = images[:int(len(images) // 3)]
        aug_images = images[int(len(images) // 3):]
        target = target.cuda(args.gpu, non_blocking=True)

        with torch.cuda.amp.autocast():
            output, text_features, logits_teacher = model(aug_images)

            weak_output = output[:int(len(output) // 2)]
            strong_output = output[int(len(output) // 2):]
            weak_output_ori = weak_output
            weak_output, selected_idx = select_confident_samples_pred(weak_output, thred=args.s1_thred)

            _, weak_preds = weak_output.detach().topk(1, 1, True, True)
            weak_preds = weak_preds.squeeze(1)

            loss = F.cross_entropy(strong_output[selected_idx], weak_preds)
            preds_extra_txt = model.logit_scale.exp() * model.extra_text_features @ text_features.t()
            loss2 = args.s1_text_loss_weight * F.cross_entropy(preds_extra_txt, model.extra_text_features_labels)
            loss3 = args.s1_div_loss_weight * (len(strong_output) / len(selected_idx)) * (div(weak_output_ori) + div(strong_output)) / 2
            loss = loss + loss2 + loss3

        optimizer.zero_grad()
        # Compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output, _, _ = model(ori_images)

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1[0], ori_images.size(0))
        top5.update(acc5[0], ori_images.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    return [top1.avg, top5.avg]


def test_time_adapt_eval(val_loader, model, optimizer, optim_state, scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # Reset model and switch to evaluate mode
    model.eval()
    with torch.no_grad():
        model.reset()
    end = time.time()
    for i, (images, target) in enumerate(val_loader):
        assert args.gpu is not None
        if isinstance(images, list):
            for k in range(len(images)):
                images[k] = images[k].cuda(args.gpu, non_blocking=True)
            image = images[0]
        else:
            if len(images.size()) > 4:
                # When using ImageNet Sampler as the dataset
                assert images.size()[0] == 1
                images = images.squeeze(0)
            images = images.cuda(args.gpu, non_blocking=True)
            image = images
        target = target.cuda(args.gpu, non_blocking=True)

        images = torch.cat(images, dim=0)

        # Reset the tunable prompt to its initial state
        if args.s2_steps > 0:
            with torch.no_grad():
                model.reset()
        optimizer.load_state_dict(optim_state)
        test_time_tuning(model, images, optimizer, scaler, args)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output, _, _ = model(image)
        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1[0], image.size(0))
        top5.update(acc5[0], image.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    return [top1.avg, top5.avg]


def select_confident_samples(logits, top):
    batch_entropy = -(logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


def test_time_tuning(model, inputs, optimizer, scaler, args):
    selected_idx = None
    for j in range(args.s2_steps):
        with torch.cuda.amp.autocast():
            output, _, _ = model(inputs)

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.s2_selection_p)

            loss = avg_entropy(output)

        optimizer.zero_grad()
        # Compute gradient and do SGD step
        scaler.scale(loss).backward()
        # Unscales the gradients of optimizer's assigned params in-place
        scaler.step(optimizer)
        scaler.update()

    return


def test(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')

    # Reset model and switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (images, target) in enumerate(val_loader):

        for k in range(len(images)):
            images[k] = images[k].cuda(args.gpu, non_blocking=True)
        images = torch.cat(images, dim=0)
        ori_images = images[:int(len(images) // 3)]
        target = target.cuda(args.gpu, non_blocking=True)

        with torch.no_grad():
            with torch.cuda.amp.autocast():
                output, _, _ = model(ori_images)

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1[0], ori_images.size(0))
        top5.update(acc5[0], ori_images.size(0))

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()

    return [top1.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Instance-to-task Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--dataset', type=str, default='I',
                        choices=['I', 'A', 'K', 'R', 'V', 'DTD', 'Flower102', 'Food101', 'Cars', 'SUN397', 'Aircraft', 'Pets', 'Caltech101', 'UCF101', 'eurosat'],
                        help='test dataset')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-p', '--print-freq', default=50, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--output_dir', default="", type=str)

    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default="a_photo_of_a", type=str, help='init tunable prompts')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('-s1_b', '--s1-batch-size', default=256, type=int, metavar='N')
    parser.add_argument('-s1_lr', '--s1-learning-rate', default=0.001, type=float, metavar='LR', dest='s1_lr')
    parser.add_argument('-s1_epoch', '--s1-epoch', default=1, type=int, metavar='N')
    parser.add_argument('-s1_thred', '--s1-thred', default=0.7, type=float, help='confidence selection threshold')
    parser.add_argument('-s1_text_loss_weight', '--s1-text-loss-weight', default=0.1, type=float, help='loss weight of text loss')
    parser.add_argument('-s1_div_loss_weight', '--s1-div-loss-weight', default=0.5, type=float, help='loss weight of diversity loss')

    parser.add_argument('--s2-views', default=64, type=int, metavar='N')
    parser.add_argument('--s2_lr', '--s2-learning-rate', default=5e-3, type=float, metavar='LR', dest='s2_lr')
    parser.add_argument('--s2-selection-p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--s2-steps', default=1, type=int)

    args = parser.parse_args()

    set_random_seed(args.seed)
    setup_logger(args.output_dir)

    print(args)
    main_worker(args)


