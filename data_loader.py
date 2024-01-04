"""
StarGAN v2
Copyright (c) 2020-present NAVER Corp.

This work is licensed under the Creative Commons Attribution-NonCommercial
4.0 International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
"""

import json
from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
from PIL import Image
import numpy as np

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder

import util.custom_transforms as ct


def listdir(dir):
# find all files with extension png or jpg in a directory dir
    fnames = list(chain(*[list(Path(dir).rglob('*.' + ext))
                          for ext in ['png', 'jpg', 'jpeg', 'JPG']]))
    return fnames


class DefaultDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples = listdir(root)
        self.samples.sort()
        self.transform = transform
        self.targets = None

    def __getitem__(self, index):
        fname = self.samples[index]
        img = Image.open(fname).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

class MultiDomainDataset(data.Dataset):
    def __init__(self, root, transform=None, train_dirs=None, min_size=(200, 200)):
        self.samples = self._make_dataset(root, train_dirs=train_dirs)
        self.transform = transform
        self.min_size = min_size

    def parse_input_folders(self, input_dirs, max_sample=-1, start_idx=-1, end_idx=-1):
        # parse input directories
        self.list_sample  = []
        with open(input_dirs, 'r') as f:
            input_dirs = f.readlines()
        for dir in input_dirs:
            dir = dir.strip()
            self.list_sample += listdir(dir)

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        if start_idx >= 0 and end_idx >= 0:     # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))
        return self.list_sample

    def _make_dataset(self, root, train_dirs=None):
        if train_dirs is not None:
            # input from folders listed in text file
            return self.parse_input_folders(train_dirs)
        else: 
            # input from folders in root generating labels from the folder structure
            domain_groups = os.listdir(root)
            fnames, labels = [], []
            for idx1, domain_group in enumerate(sorted(domain_groups)):
                domains = os.listdir(os.path.join(root, domain_group))
                for idx2, domain in enumerate(sorted(domains)):
                    # print('domain: {}'.format(domain))
                    label = [-1] * len(domain_groups)
                    class_dir = os.path.join(root, domain_group, domain)
                    cls_fnames = listdir(class_dir)
                    fnames += cls_fnames
                    # get main label from folder structure
                    label[idx1] = idx2
                    labels += [label] * len(cls_fnames)
                # print('fnames: {} labels: {}'.format(fnames, labels))
            return fnames, labels

    def __getitem__(self, index):
        fname = self.samples[index]
        # label = self.targets[index]
        # print('label: {}'.format(label))
        img = Image.open(fname).convert('RGB')

        # Check image size and discard small images
        # if img.size[0] < self.min_size[0] or img.size[1] < self.min_size[1]:
        #     return
        
        parent, name = str(fname.parent), fname.name
        sample = Munch(img=img)
        domain_path = os.path.join(parent.replace('images', 'labels'), name.replace('.jpg', '.npy'))
        if os.path.exists(domain_path): 
            domain = np.load(domain_path)
            assert(domain.size != 0)
            sample.domain = domain
        else: 
            sample.domain = np.zeros(0)
        mask_path = os.path.join(parent.replace('images', 'seg_labels'), name.replace('.jpg', '.png'))
        if os.path.exists(mask_path):
            sample.seg_mask = Image.open(mask_path).convert('L')
        else:
            sample.seg_mask  = []

        if self.transform is not None:
            img = self.transform(sample)

        sample.domain = torch.tensor(sample.domain)
    
        return sample

    def __len__(self):
        return len(self.samples)


class ReferenceDataset(data.Dataset):
    def __init__(self, root, transform=None):
        self.samples, self.targets = self._make_dataset(root)
        self.transform = transform

    def _make_dataset(self, root):
        domains = os.listdir(root)
        fnames, fnames2, labels = [], [], []
        for idx, domain in enumerate(sorted(domains)):
            class_dir = os.path.join(root, domain)
            cls_fnames = listdir(class_dir)
            fnames += cls_fnames
            fnames2 += random.shuffle(cls_fnames)
            labels += [idx] * len(cls_fnames)
        return list(zip(fnames, fnames2)), labels

    def __getitem__(self, index):
        fname, fname2 = self.samples[index]
        label = self.targets[index]
        img = Image.open(fname).convert('RGB')
        img2 = Image.open(fname2).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
            img2 = self.transform(img2)
        return img, img2, label

    def __len__(self):
        return len(self.targets)


def _make_balanced_sampler(labels):
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def _make_mult_domain_balanced_sampler(labels):
    print('labels: {}'.format(labels))
    class_counts = np.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))

def get_train_loader(root, which='source', img_size=(256, 256),
                     batch_size=8, prob=0.5, num_workers=4, train_dirs=None, imagenet_normalize=True, max_scale=2.0):
    print('Preparing DataLoader to fetch %s images '
          'during the training phase...' % which)

    if imagenet_normalize:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform= transforms.Compose([
        ct.RandomScale(max_scale, img_size[0]),
        ct.RandomCrop(img_size),
        ct.HorizontalFlip(),
        ct.ToTensor(),
        ct.Normalize(mean=mean,
                std=std),
        ])
    if which == 'source':
        dataset = ImageFolder(root, transform)
    elif which == 'reference':
        dataset = ReferenceDataset(root, transform)
    elif which == 'multi':
        dataset = MultiDomainDataset(root, transform, train_dirs=train_dirs, min_size=[max_scale*img_size[0], max_scale*img_size[1]])
    else:
        raise NotImplementedError

    # sampler = _make_balanced_sampler(dataset.targets)
    # sampler = _make_mult_domain_balanced_sampler(dataset.targets)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                        #    sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)


def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    print('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
        height, width = 299, 299
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        height, width = img_size, img_size
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.Resize([height, width]),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = DefaultDataset(root, transform=transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=drop_last)


def get_test_loader(root, img_size=256, batch_size=32,
                    shuffle=True, num_workers=4):
    print('Preparing DataLoader for the generation phase...')
    transform = transforms.Compose([
        transforms.Resize([img_size, img_size]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5]),
    ])

    dataset = ImageFolder(root, transform)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=shuffle,
                           num_workers=num_workers,
                           pin_memory=True)


class InputFetcher:
    def __init__(self, loader, loader_ref=None, latent_dim=16, mode='', num_domains=2):
        self.loader = loader
        self.loader_ref = loader_ref
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.num_domains = num_domains

    def _fetch_inputs(self):
        try:
            x = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x = next(self.iter)
        return x

    def _fetch_refs(self):
        try:
            x, x2, y = next(self.iter_ref)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x, x2, y = next(self.iter_ref)
        return x, x2, y

    def generate_labels(self, y_src, y_trg, num_domains):
        # create one-hot label vectors
        y_src = torch.zeros(y_src.size(0), num_domains).scatter_(1, y_src.view(-1, 1), 1)
        y_trg = torch.zeros(y_trg.size(0), num_domains).scatter_(1, y_trg.view(-1, 1), 1)
        return y_src.to(self.device), y_trg.to(self.device)

    def __next__(self):
        x = self._fetch_inputs()
        if self.mode == 'train':
            y_trg = (torch.rand(1)*self.num_domains).int().to(self.device)
            z_trg = torch.randn(x.img.size(0), self.latent_dim)
            z_trg2 = torch.randn(x.img.size(0), self.latent_dim)
            inputs = Munch(x_src=x.img, y_src=x.domain, seg = x.seg_mask, y_trg=y_trg,
                           z_trg=z_trg, z_trg2=z_trg2)
        # elif self.mode == 'val':
        #     x_ref, seg, y_ref = self._fetch_inputs()
        #     inputs = Munch(x_src=x, y_src=x,
        #                    x_ref=x_ref, y_ref=y_ref, seg=seg)
        elif self.mode == 'test':
            inputs = Munch(x=x, z=z_trg)
        else:
            raise NotImplementedError

        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})