import logging
import json
from pathlib import Path
from itertools import chain
import os
import random

from munch import Munch
import numpy as np
from PIL import Image, UnidentifiedImageError

import torch
from torch.utils import data
from torch.utils.data.sampler import WeightedRandomSampler
from torchvision import transforms
from torchvision.datasets import ImageFolder
import time
import util.custom_transforms as ct

from utils import domain_to_onehot, get_domain_indexes

Image.MAX_IMAGE_PIXELS = 103782392

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
        try:
            img = Image.open(fname).convert('RGB')
        except UnidentifiedImageError:
            img = Image.new('RGB', (384, 384))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)

def parse_input_folders(input_file, max_sample=-1, start_idx=-1, end_idx=-1):
    # parse input directories
    list_sample  = []
    if isinstance(input_file, str):
        with open(input_file, 'r') as f:
            input_dirs = f.readlines()
    elif isinstance(input_file, list):
        input_dirs = input_file
    else:
        raise NotImplementedError('Input file type not supported')
         
    for dir in input_dirs:
        dir = dir.strip()
        list_sample += listdir(dir)

    if max_sample > 0:
        list_sample = list_sample[0:max_sample]
    if start_idx >= 0 and end_idx >= 0:     # divide file list
        list_sample = list_sample[start_idx:end_idx]

    return list_sample


class MultiDomainDataset(data.Dataset):
    def __init__(self, train_list=None, transform=None, target_domains=[]):
        self.imgs, self.domains = [], []
        self.target_domains = target_domains
        self._make_dataset(train_list=train_list)
        self.transform = transform

    def _make_dataset(self, train_list=None):
        if train_list is not None:
            self.imgs = parse_input_folders(train_list)     
            for img_path in self.imgs:       
            # input from folders listed in text file
                parent, name = str(img_path.parent), img_path.name
                domain_path = os.path.join(parent.replace('images', 'domain_labels'), 
                    name.replace('.jpg', '.npy')).replace('.png', '.npy')
                if os.path.exists(domain_path): 
                    domain = np.load(domain_path)
                    assert(domain.size > 1)
                    idxs, _ = get_domain_indexes(self.target_domains)
                    self.domains.append(domain[idxs])
                else: 
                    self.domains.append(np.zeros(len(self.target_domains)) 
           )
        logging.info('{} samples found in: {}'.format(len(self.imgs), train_list))                    
        return
    def __getitem__(self, index):
        img_path = self.imgs[index]
        domain = self.domains[index]
        try:
            img = Image.open(img_path).convert('RGB')
        except UnidentifiedImageError:
            img = Image.new('RGB', (384, 384))
        sample = Munch(img=img, domain=domain)
        parent, name = str(img_path.parent), img_path.name        
        seg_path = os.path.join(parent.replace('images', 'sem_labelss'), name.replace('.jpg', '.png'))
        if os.path.exists(seg_path):
            try:
                sem_labels = Image.open(seg_path).convert('L')
            except UnidentifiedImageError:
                sem_labels = Image.new('L', (352, 352))
        else:
            sem_labels  = Image.new('L', (352, 352))
        sample.sem_labels = sem_labels
        if self.transform is not None:
            sample = self.transform(sample)
        sample.domain = torch.tensor(sample.domain)
        return sample
    
    def __len__(self):
        return len(self.domains)

class ReferenceDataset(data.Dataset):
    def __init__(self, ref_list, transform=None, target_domains=[]):
        self.ref_samples, self.ref_domains = [], []
        self.transform = transform
        self.target_domains = target_domains
        self._make_dataset(ref_list)

    def _make_dataset(self, ref_list=None):
        for folder in ref_list:
            self.ref_samples += parse_input_folders(ref_list)
            _domain = folder.split('/')[-1]
            self.ref_domains += [domain_to_onehot(_domain, self.target_domains)] * len(self.ref_samples)
        assert self.ref_samples != [], 'No reference images found'         
        logging.info('{} samples found in: {}'.format(len(self.ref_samples), ref_list))                    
        return self.ref_samples, self.ref_domains

    def __getitem__(self, index, d_src=None):
        # make sure that the reference image is not from the same domain as the source image
        if d_src is not None:
            while torch.equal(torch.argmax(d_src), torch.argmax(self.ref_domains[index])):
                index = random.randint(0, len(self.samples) - 1)
                file_path = self.ref_samples[index]
                domain = self.ref_domains[index]
        else: 
            file_path = self.ref_samples[index]
            domain = self.ref_domains[index]
        try:
            img = Image.open(file_path).convert('RGB')
        except UnidentifiedImageError:
            img = Image.new('RGB', (384, 384))
        if self.transform is not None:
            img = self.transform(img)       
        sample = Munch(img=img, domain=domain)
        return sample

    def __len__(self):
        return len(self.ref_samples)


def _make_balanced_sampler(labels):
    class_counts = torch.bincount(labels)
    class_weights = 1. / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(img_size: (int, int)=(256, 256),
                     batch_size: int=8, prob: float=0.5, num_workers: int=8, train_list: str=None, imagenet_normalize: bool=True, max_scale: float=2.0, max_n_bbox=4, target_domains=[]):
    logging.info('Preparing DataLoader to fetch training images '
          'during the training phase...')

    if imagenet_normalize == True:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(1.0, 2.0)),
        transforms.RandomHorizontalFlip(p=prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    transform_custom = transforms.Compose([
        ct.RandomScale(img_size),
        ct.RandomCrop(img_size),
        ct.HorizontalFlip(),
        ct.ToTensor(),        
        ct.Normalize(mean=mean,
                std=std),
        ct.SegMaskToBBoxes([1, 7, 14], n_bbox=max_n_bbox),
        ct.SegMaskToPatches(8, 0.8)
        ])
    dataset = MultiDomainDataset(train_list, transform_custom, target_domains=target_domains)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True, # use shuffle or sample
                        #    sampler=sampler,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)

def get_ref_loader(img_size: (int, int) = (256, 256),
                   batch_size: int = 8, prob: float = 0.5, num_workers: int = 8, ref_list: str = None,
                   imagenet_normalize: bool = True, max_scale: float = 2.0, target_domains=[]):
    logging.info('Preparing DataLoader to fetch reference images during the training phase...')

    if imagenet_normalize:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    else:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]

    transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=prob),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset = ReferenceDataset(ref_list, transform, target_domains=target_domains)
    return data.DataLoader(dataset=dataset,
                           batch_size=batch_size,
                           shuffle=True,
                           num_workers=num_workers,
                           pin_memory=True,
                           drop_last=True)

def get_eval_loader(root, img_size=256, batch_size=32,
                    imagenet_normalize=True, shuffle=True,
                    num_workers=4, drop_last=False):
    logging.info('Preparing DataLoader for the evaluation phase...')
    if imagenet_normalize:
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
    logging.info('Preparing DataLoader for the generation phase...')
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


class InputProvider:
    def __init__(self, loader, latent_dim=16, mode='', num_domains=12):
        self.loader = loader
        self.iter = iter(self.loader)
        self.latent_dim = latent_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.num_domains = num_domains

        # self.iter = iter(self.loader)
        # self.iter_ref = iter(self.loader_ref)

    def _fetch_inputs(self):
        try:
            x = next(self.iter)
        except (AttributeError, StopIteration):
            self.iter = iter(self.loader)
            x = next(self.iter)
        return x

    def __next__(self):
        # return
        sample = self._fetch_inputs()
        # ref = self._fetch_refs()        
        if self.mode == 'train':
            lat_trg = torch.randn(sample.img.size(0), self.latent_dim)
            lat_trg2 = torch.randn(sample.img.size(0), self.latent_dim)
            inputs = Munch(img_src=sample.img, d_src=sample.domain, seg = sample.sem_labels, lat_trg=lat_trg, lat_trg2=lat_trg2, bbox=sample.bboxes)
        elif self.mode == 'val':
            lat_trg = torch.randn(sample.img.size(0), self.latent_dim)
            inputs = Munch(img_src=sample.img, d_src=sample.domain, seg=sample.sem_labels, lat_trg=lat_trg)
        elif self.mode == 'test':
            lat_trg = torch.randn(sample.size(0), self.latent_dim)
            inputs = Munch(img_src=sample.img, d_src=sample.domain, seg=sample.sem_labels, lat_trg=lat_trg)
        else:
            raise NotImplementedError
        #TODO make sure not to move to GPU twice
        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})
    
class RefProvider:
    def __init__(self, loader_ref, mode=''):
        self.loader_ref = loader_ref
        # self.iter_ref = iter(self.loader_ref)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode = mode

    def _fetch_refs(self, d_src=None):
        try:
            x = next(self.iter_ref, d_src)
        except (AttributeError, StopIteration):
            self.iter_ref = iter(self.loader_ref)
            x = next(self.iter_ref, d_src)
        return x


    def __next__(self, d_src=None):
        ref = self._fetch_refs(d_src)
        while ref.img is None:
            ref = self._fetch_refs(d_src)
        if self.mode == 'train':
            inputs = Munch(img_ref = ref.img, d_trg=ref.domain)
        elif self.mode == 'val':
            inputs = Munch(img_ref=ref.img, d_trg=ref.domain)
        elif self.mode == 'test':
            inputs = Munch(d_trg=ref.domain)
        else:
            raise NotImplementedError
        #TODO make sure not to move to GPU twice
        return Munch({k: v.to(self.device)
                      for k, v in inputs.items()})    