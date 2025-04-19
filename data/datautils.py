import os
from typing import Tuple
from PIL import Image
import numpy as np

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data.hoi_dataset import BongardDataset
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

from data.fewshot_datasets import *
import data.augmix_ops as augmentations

# ID_to_DIRNAME={
#     'I': 'ImageNet',
#     'A': 'imagenet-a',
#     'K': 'ImageNet-Sketch',
#     'R': 'imagenet-r',
#     'V': 'imagenetv2-matched-frequency-format-val',
#     'flower102': 'Flower102',
#     'dtd': 'DTD',
#     'pets': 'OxfordPets',
#     'cars': 'StanfordCars',
#     'ucf101': 'UCF101',
#     'caltech101': 'Caltech101',
#     'food101': 'Food101',
#     'sun397': 'SUN397',
#     'aircraft': 'fgvc_aircraft',
#     'eurosat': 'eurosat'
# }

ID_to_DIRNAME={
    'I': 'imagenet',
    'A': 'imagenet-adversarial/imagenet-a',
    'K': 'imagenet-sketch/images',
    'R': 'imagenet-rendition/imagenet-r',
    'V': 'imagenetv2/imagenetv2-matched-frequency-format-val',
    'flower102': 'oxford_flowers',
    'dtd': 'dtd',
    'pets': 'oxford_pets',
    'cars': 'stanford_cars',
    'ucf101': 'ucf101',
    'caltech101': 'caltech-101',
    'food101': 'food-101',
    'sun397': 'sun397',
    'aircraft': 'fgvc_aircraft',
    'eurosat': 'eurosat'
}

def build_dataset(set_id, transform, data_root, mode='test', n_shot=None, split="all", bongard_anno=False):
    if set_id == 'I':
        # ImageNet validation set
        testdir = os.path.join(os.path.join(data_root, ID_to_DIRNAME[set_id]), 'val')
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in ['A', 'K', 'R', 'V']:
        testdir = os.path.join(data_root, ID_to_DIRNAME[set_id])
        testset = datasets.ImageFolder(testdir, transform=transform)
    elif set_id in fewshot_datasets:
        if mode == 'train' and n_shot:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode, n_shot=n_shot)
        else:
            testset = build_fewshot_dataset(set_id, os.path.join(data_root, ID_to_DIRNAME[set_id.lower()]), transform, mode=mode)
    elif set_id == 'bongard':
        assert isinstance(transform, Tuple)
        base_transform, query_transform = transform
        testset = BongardDataset(data_root, split, mode, base_transform, query_transform, bongard_anno)
    else:
        raise NotImplementedError
        
    return testset


# AugMix Transforms
def get_preaugment():
    return transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ])

def augmix(image, preprocess, aug_list, severity=1):
    preaugment = get_preaugment()
    x_orig = preaugment(image)
    x_processed = preprocess(x_orig)
    if len(aug_list) == 0:
        return x_processed
    w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
    m = np.float32(np.random.beta(1.0, 1.0))

    mix = torch.zeros_like(x_processed)
    for i in range(3):
        x_aug = x_orig.copy()
        for _ in range(np.random.randint(1, 4)):
            x_aug = np.random.choice(aug_list)(x_aug, severity)
        mix += w[i] * preprocess(x_aug)
    mix = m * x_processed + (1 - m) * mix
    return mix


class AugMixAugmenter(object):
    def __init__(self, base_transform, preprocess, n_views=2, augmix=False, 
                    severity=1):
        self.base_transform = base_transform
        self.preprocess = preprocess
        self.n_views = n_views
        if augmix:
            self.aug_list = augmentations.augmentations
        else:
            self.aug_list = []
        self.severity = severity
        
    def __call__(self, x):
        image = self.preprocess(self.base_transform(x))
        views = [augmix(x, self.preprocess, self.aug_list, self.severity) for _ in range(self.n_views)]
        return [image] + views


class ImageNetAugmenter(object):
    def __init__(self, base_transform, postprocess, n_views=2):
        self.n_views = n_views

        self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.base_transform = base_transform
        self.postprocess = postprocess

    def __call__(self, x):
        image = self.postprocess(self.base_transform(x))
        views = [self.postprocess(self.flip(self.randomresizedcrop(x))) for _ in range(self.n_views)]
        return [image] + views


# def stongaug(image, preprocess, aug_list, severity=1):
#     preaugment = get_preaugment()
#     x_orig = preaugment(image)
#     x_processed = preprocess(x_orig)
#     if len(aug_list) == 0:
#         return x_processed
#     w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
#     m = np.float32(np.random.beta(1.0, 1.0))
#
#     mix = torch.zeros_like(x_processed)
#     for i in range(3):
#         x_aug = x_orig.copy()
#         for _ in range(np.random.randint(1, 4)):
#             x_aug = np.random.choice(aug_list)(x_aug, severity)
#         mix += w[i] * preprocess(x_aug)
#     mix = m * x_processed + (1 - m) * mix
#     return mix
#
#
# class WeakStrongAugmenter(object):
#     def __init__(self, base_transform, postprocess):
#
#         self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1), interpolation=BICUBIC)
#         self.flip = transforms.RandomHorizontalFlip(p=0.5)
#         self.base_transform = base_transform
#         self.postprocess = postprocess
#         self.cutout = Cutout(1, 56)
#
#
#     def __call__(self, x):
#         ori_image = self.postprocess(self.base_transform(x))
#         weak_aug_image = self.postprocess(self.flip(self.randomresizedcrop(x)))
#         strong_aug_image = stongaug(x, self.postprocess, augmentations.augmentations, 4)
#         p = np.random.random()
#         if p < 0.5:
#             strong_aug_image = self.cutout(self.flip(self.randomresizedcrop(self.postprocess(x))))
#         return [ori_image, weak_aug_image, strong_aug_image]


import torch
import numpy as np


class Cutout(object):
    """Randomly mask out one or more patches from an image.
    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length):
        self.n_holes = n_holes
        self.length = length

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img


class WeakStrongAugmenter(object):
    def __init__(self, base_transform, postprocess):
        self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                                              interpolation=BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.base_transform = base_transform
        self.postprocess = postprocess
        self.cutout = Cutout(1, 56)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.aug_list = augmentations.augmentations

    def __call__(self, x):
        ori_image = self.postprocess(self.base_transform(x))
        weak_aug_image = self.postprocess(self.flip(self.randomresizedcrop(x)))
        strong_aug_image = self.stongaug(x)
        p = np.random.random()
        if p < 0.5:
            strong_aug_image = self.normalize(self.cutout(self.to_tensor(self.flip(self.randomresizedcrop(x)))))
        return [ori_image, weak_aug_image, strong_aug_image]

    def stongaug(self, image, severity=4):
        # preaugment = get_preaugment()
        # x_orig = preaugment(image)
        x_orig = self.flip(self.randomresizedcrop(image))
        x_orig_tensor = self.to_tensor(x_orig)

        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_orig_tensor)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.aug_list)(x_aug, severity)
            mix += w[i] * self.to_tensor(x_aug) #preprocess(x_aug)
        mix = m * x_orig_tensor + (1 - m) * mix
        mix = self.normalize(mix)
        return mix

class WeakStrongAugmenter2(object):
    def __init__(self, base_transform, postprocess):
        self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                                              interpolation=BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.base_transform = base_transform
        self.postprocess = postprocess
        self.cutout = Cutout(1, 56)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.aug_list = augmentations.augmentations

    def __call__(self, x):
        ori_image = self.postprocess(self.base_transform(x))
        weak_aug_image = self.postprocess(self.flip(self.randomresizedcrop(x)))
        strong_aug_image1 = self.stongaug(x)
        p = np.random.random()
        if p < 0.5:
            strong_aug_image1 = self.normalize(self.cutout(self.to_tensor(self.flip(self.randomresizedcrop(x)))))
        strong_aug_image2 = self.stongaug(x)
        p = np.random.random()
        if p < 0.5:
            strong_aug_image2 = self.normalize(self.cutout(self.to_tensor(self.flip(self.randomresizedcrop(x)))))
        return [ori_image, weak_aug_image, strong_aug_image1, strong_aug_image2]

    def stongaug(self, image, severity=4):
        # preaugment = get_preaugment()
        # x_orig = preaugment(image)
        x_orig = self.flip(self.randomresizedcrop(image))
        x_orig_tensor = self.to_tensor(x_orig)

        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_orig_tensor)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.aug_list)(x_aug, severity)
            mix += w[i] * self.to_tensor(x_aug) #preprocess(x_aug)
        mix = m * x_orig_tensor + (1 - m) * mix
        mix = self.normalize(mix)
        return mix

class WeakWeakStrongAugmenter(object):
    def __init__(self, base_transform, postprocess):
        self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                                              interpolation=BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.base_transform = base_transform
        self.postprocess = postprocess
        self.cutout = Cutout(1, 56)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.aug_list = augmentations.augmentations

    def __call__(self, x):
        ori_image = self.postprocess(self.base_transform(x))
        weak_aug_image = self.postprocess(self.flip(self.randomresizedcrop(x)))
        weak_aug_image2 = self.postprocess(self.flip(self.randomresizedcrop(x)))
        strong_aug_image = self.stongaug(x)
        p = np.random.random()
        if p < 0.5:
            strong_aug_image = self.normalize(self.cutout(self.to_tensor(self.flip(self.randomresizedcrop(x)))))
        return [ori_image, weak_aug_image, weak_aug_image2, strong_aug_image]

    def stongaug(self, image, severity=4):
        # preaugment = get_preaugment()
        # x_orig = preaugment(image)
        x_orig = self.flip(self.randomresizedcrop(image))
        x_orig_tensor = self.to_tensor(x_orig)

        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_orig_tensor)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.aug_list)(x_aug, severity)
            mix += w[i] * self.to_tensor(x_aug) #preprocess(x_aug)
        mix = m * x_orig_tensor + (1 - m) * mix
        mix = self.normalize(mix)
        return mix

class OriStrongAugmenter(object):
    def __init__(self, base_transform, postprocess):
        self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                                              interpolation=BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.base_transform = base_transform
        self.postprocess = postprocess
        self.cutout = Cutout(1, 56)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.aug_list = augmentations.augmentations

    def __call__(self, x):
        ori_image = self.postprocess(self.base_transform(x))
        weak_aug_image = self.postprocess(self.base_transform(x))
        strong_aug_image = self.stongaug(x)
        p = np.random.random()
        if p < 0.5:
            strong_aug_image = self.normalize(self.cutout(self.to_tensor(self.flip(self.randomresizedcrop(x)))))
        return [ori_image, weak_aug_image, strong_aug_image]

    def stongaug(self, image, severity=4):
        # preaugment = get_preaugment()
        # x_orig = preaugment(image)
        x_orig = self.flip(self.randomresizedcrop(image))
        x_orig_tensor = self.to_tensor(x_orig)

        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_orig_tensor)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.aug_list)(x_aug, severity)
            mix += w[i] * self.to_tensor(x_aug) #preprocess(x_aug)
        mix = m * x_orig_tensor + (1 - m) * mix
        mix = self.normalize(mix)
        return mix


class WeakBaseAugmenter(object):
    def __init__(self, base_transform, postprocess):
        self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                                              interpolation=BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.base_transform = base_transform
        self.postprocess = postprocess
        self.cutout = Cutout(1, 56)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.aug_list = augmentations.augmentations

    def __call__(self, x):
        ori_image = self.postprocess(self.base_transform(x))
        weak_aug_image = self.postprocess(self.flip(self.randomresizedcrop(x)))
        strong_aug_image = self.postprocess(self.flip(self.randomresizedcrop(x)))
        return [ori_image, weak_aug_image, strong_aug_image]

    def stongaug(self, image, severity=4):
        # preaugment = get_preaugment()
        # x_orig = preaugment(image)
        x_orig = self.flip(self.randomresizedcrop(image))
        x_orig_tensor = self.to_tensor(x_orig)

        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_orig_tensor)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.aug_list)(x_aug, severity)
            mix += w[i] * self.to_tensor(x_aug) #preprocess(x_aug)
        mix = m * x_orig_tensor + (1 - m) * mix
        mix = self.normalize(mix)
        return mix


class WeakRandAugmenter(object):
    def __init__(self, base_transform, postprocess):
        self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                                              interpolation=BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.base_transform = base_transform
        self.postprocess = postprocess
        self.cutout = Cutout(1, 56)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.aug_list = augmentations.augmentations

    def __call__(self, x):
        ori_image = self.postprocess(self.base_transform(x))
        weak_aug_image = self.postprocess(self.flip(self.randomresizedcrop(x)))
        strong_aug_image = self.stongaug(x)
        return [ori_image, weak_aug_image, strong_aug_image]

    def stongaug(self, image, severity=4):
        # preaugment = get_preaugment()
        # x_orig = preaugment(image)
        x_orig = self.flip(self.randomresizedcrop(image))
        x_orig_tensor = self.to_tensor(x_orig)

        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_orig_tensor)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.aug_list)(x_aug, severity)
            mix += w[i] * self.to_tensor(x_aug) #preprocess(x_aug)
        mix = m * x_orig_tensor + (1 - m) * mix
        mix = self.normalize(mix)
        return mix

class WeakCutoutAugmenter(object):
    def __init__(self, base_transform, postprocess):
        self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                                              interpolation=BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.base_transform = base_transform
        self.postprocess = postprocess
        self.cutout = Cutout(1, 56)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.aug_list = augmentations.augmentations

    def __call__(self, x):
        ori_image = self.postprocess(self.base_transform(x))
        weak_aug_image = self.postprocess(self.flip(self.randomresizedcrop(x)))
        strong_aug_image = self.normalize(self.cutout(self.to_tensor(self.flip(self.randomresizedcrop(x)))))
        return [ori_image, weak_aug_image, strong_aug_image]

    def stongaug(self, image, severity=4):
        # preaugment = get_preaugment()
        # x_orig = preaugment(image)
        x_orig = self.flip(self.randomresizedcrop(image))
        x_orig_tensor = self.to_tensor(x_orig)

        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_orig_tensor)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.aug_list)(x_aug, severity)
            mix += w[i] * self.to_tensor(x_aug) #preprocess(x_aug)
        mix = m * x_orig_tensor + (1 - m) * mix
        mix = self.normalize(mix)
        return mix

class StrongBaseAugmenter(object):
    def __init__(self, base_transform, postprocess):
        self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                                              interpolation=BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.base_transform = base_transform
        self.postprocess = postprocess
        self.cutout = Cutout(1, 56)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.aug_list = augmentations.augmentations

    def __call__(self, x):
        ori_image = self.postprocess(self.base_transform(x))
        weak_aug_image = self.postprocess(self.flip(self.randomresizedcrop(x)))
        strong_aug_image = self.stongaug(x)
        p = np.random.random()
        if p < 0.5:
            strong_aug_image = self.normalize(self.cutout(self.to_tensor(self.flip(self.randomresizedcrop(x)))))
        return [ori_image, strong_aug_image, weak_aug_image]

    def stongaug(self, image, severity=4):
        # preaugment = get_preaugment()
        # x_orig = preaugment(image)
        x_orig = self.flip(self.randomresizedcrop(image))
        x_orig_tensor = self.to_tensor(x_orig)

        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_orig_tensor)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.aug_list)(x_aug, severity)
            mix += w[i] * self.to_tensor(x_aug) #preprocess(x_aug)
        mix = m * x_orig_tensor + (1 - m) * mix
        mix = self.normalize(mix)
        return mix


class StrongStrongAugmenter(object):
    def __init__(self, base_transform, postprocess):
        self.randomresizedcrop = transforms.RandomResizedCrop(size=224, scale=(0.5, 1),
                                                              interpolation=BICUBIC)
        self.flip = transforms.RandomHorizontalFlip(p=0.5)
        self.base_transform = base_transform
        self.postprocess = postprocess
        self.cutout = Cutout(1, 56)

        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                              std=[0.26862954, 0.26130258, 0.27577711])

        self.aug_list = augmentations.augmentations

    def __call__(self, x):
        ori_image = self.postprocess(self.base_transform(x))
        weak_aug_image = self.stongaug(x)
        p = np.random.random()
        if p < 0.5:
            weak_aug_image = self.normalize(self.cutout(self.to_tensor(self.flip(self.randomresizedcrop(x)))))
        strong_aug_image = self.stongaug(x)
        p = np.random.random()
        if p < 0.5:
            strong_aug_image = self.normalize(self.cutout(self.to_tensor(self.flip(self.randomresizedcrop(x)))))
        return [ori_image, weak_aug_image, strong_aug_image]

    def stongaug(self, image, severity=4):
        # preaugment = get_preaugment()
        # x_orig = preaugment(image)
        x_orig = self.flip(self.randomresizedcrop(image))
        x_orig_tensor = self.to_tensor(x_orig)

        w = np.float32(np.random.dirichlet([1.0, 1.0, 1.0]))
        m = np.float32(np.random.beta(1.0, 1.0))

        mix = torch.zeros_like(x_orig_tensor)
        for i in range(3):
            x_aug = x_orig.copy()
            for _ in range(np.random.randint(1, 4)):
                x_aug = np.random.choice(self.aug_list)(x_aug, severity)
            mix += w[i] * self.to_tensor(x_aug) #preprocess(x_aug)
        mix = m * x_orig_tensor + (1 - m) * mix
        mix = self.normalize(mix)
        return mix