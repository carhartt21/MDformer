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
    fnames = list(
        chain(
            *[
                list(Path(dir).rglob("*." + ext))
                for ext in ["png", "jpg", "jpeg", "JPG"]
            ]
        )
    )
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
            img = Image.open(fname).convert("RGB")
        except UnidentifiedImageError:
            img = Image.new("RGB", (384, 384))
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)


def parse_input_folders(input_file, max_sample=-1, start_idx=-1, end_idx=-1):
    # parse input directories
    list_sample = []
    if isinstance(input_file, str):
        with open(input_file, "r") as f:
            input_dirs = f.readlines()
    elif isinstance(input_file, list):
        input_dirs = input_file
    else:
        raise NotImplementedError("Input file type not supported")

    for dir in input_dirs:
        dir = dir.strip()
        list_sample += listdir(dir)

    if max_sample > 0:
        list_sample = list_sample[0:max_sample]
    if start_idx >= 0 and end_idx >= 0:  # divide file list
        list_sample = list_sample[start_idx:end_idx]

    return list_sample


class MultiDomainDataset(data.Dataset):
    def __init__(
        self,
        train_list=None,
        ref_list=None,
        transform=None,
        target_domain_names=[],
        max_sample=-1,
        input_size=(320, 320),
    ):
        self.imgs, self.src_domains = [], []
        self.ref_samples, self.trg_domains = [], []
        self.train_list = train_list
        self.ref_list = ref_list
        self.target_domain_names = target_domain_names
        self.transform = transform
        self.max_sample = max_sample
        self.input_size = input_size
        self._make_dataset()
        self._make_ref_dataset()

    def _make_dataset(self):
        for domain_folder_path in self.train_list:
            # convert domain to one-hot vector for all samples in the folder
            _samples = parse_input_folders(
                [domain_folder_path], max_sample=self.max_sample
            )
            domain_name = domain_folder_path.split("/")[-1]
            self.src_domains += [
                domain_to_onehot(domain_name, self.target_domain_names)
            ] * len(_samples)
            logging.info(f">> {len(_samples):,} samples found in: {domain_name}")
            self.imgs += _samples
        assert self.imgs != [], ">> No training images found"
        logging.info(f">>>> Total: {len(self.imgs):,} training samples found")
        return self.imgs, self.src_domains

    def _make_ref_dataset(self):
        for domain_folder_path in self.ref_list:
            # convert domain to one-hot vector for all samples in the folder
            _ref_samples = parse_input_folders(
                [domain_folder_path], max_sample=self.max_sample
            )
            domain_name = domain_folder_path.split("/")[-1]
            self.trg_domains += [
                domain_to_onehot(domain_name, self.target_domain_names)
            ] * len(_ref_samples)
            logging.info(f">> {len(_ref_samples):,} samples found in: {domain_name}")
            self.ref_samples += _ref_samples
        assert self.ref_samples != [], ">> No reference images found"
        logging.info(f">>>> Total: {len(self.ref_samples):,} reference samples found")
        return self.ref_samples, self.trg_domains

    def __getitem__(self, index):
        img_path = self.imgs[index]
        src_domain = self.src_domains[index]
        try:
            rand_idx = random.randint(0, len(self.trg_domains) - 1)
            trg_domain = self.trg_domains[rand_idx]
            while torch.argmax(src_domain) == torch.argmax(trg_domain):
                rand_idx = random.randint(0, len(self.trg_domains) - 1)
                trg_domain = self.trg_domains[rand_idx]
        except IndexError:
            trg_domain = self.trg_domains[rand_idx]
        try:
            img = Image.open(img_path).convert("RGB")
            ref_img = Image.open(self.ref_samples[rand_idx]).convert("RGB")
        except UnidentifiedImageError:
            img = Image.new("RGB", self.input_size)
        sample = Munch(
            img=img, src_domain=src_domain, trg_domain=trg_domain, ref_img=ref_img
        )
        parent, name = img_path.parent, img_path.name
        _dir, _domain = os.path.split(parent)
        seg_path = os.path.join(
            str(_dir).replace("images", "seg_masks"), name.replace(".jpg", ".png")
        )
        if os.path.exists(seg_path):
            try:
                seg_masks = Image.open(seg_path).convert("L")
            except UnidentifiedImageError:
                seg_masks = Image.new("L", img.size)
        else:
            logging.error(">> No semantic labels found for: {}".format(seg_path))
            seg_masks = Image.new("L", img.size)
        sample.seg_masks = seg_masks
        if self.transform is not None:
            sample = self.transform(sample)
        # sample.domain = torch.tensor(sample.domain)
        return sample

    def __len__(self):
        return len(self.imgs)


class ReferenceDataset(data.Dataset):
    def __init__(
        self,
        ref_list,
        transform=None,
        target_domain_names=[],
        max_sample=-1,
        input_size=(320, 320),
    ):
        self.ref_samples, self.ref_domains = [], []
        self.ref_list = ref_list
        self.transform = transform
        # target domains are the domains that we want to transfer to
        self.target_domain_names = target_domain_names
        self.max_sample = max_sample
        self.input_size = input_size
        self._make_dataset()

    def _make_dataset(self):
        for domain_folder_path in self.ref_list:
            # convert domain to one-hot vector for all samples in the folder
            _ref_samples = parse_input_folders(
                [domain_folder_path], max_sample=self.max_sample
            )
            domain_name = domain_folder_path.split("/")[-1]
            self.ref_domains += [
                domain_to_onehot(domain_name, self.target_domain_names)
            ] * len(_ref_samples)
            logging.info(
                ">> {} samples found in: {}".format(len(_ref_samples), domain_name)
            )
            self.ref_samples += _ref_samples
        assert self.ref_samples != [], ">> No reference images found"
        logging.info(f">>>> Total: {len(self.ref_samples):,} reference samples found")
        return self.ref_samples, self.ref_domains

    def __getitem__(self, index, d_src=None):
        # make sure that the reference image is not from the same domain as the source image
        # TODO: doesn't work yet because __getitem__ passes only one value

        file_path = self.ref_samples[index]
        domain = self.ref_domains[index]
        try:
            img = Image.open(file_path).convert("RGB")
        except UnidentifiedImageError:
            img = Image.new("RGB", self.input_size)
        if self.transform is not None:
            img = self.transform(img)
        sample = Munch(img=img, domain=domain)
        return sample

    def __len__(self):
        return len(self.ref_samples)


class TestDataset(data.Dataset):
    def __init__(
        self,
        test_dir="",
        transform=None,
        target_domain_names=[],
        max_sample=-1,
        input_size=(320, 320),
    ):
        self.imgs, self.seg_masks = [], []
        self.test_dir = test_dir
        self.target_domain_names = target_domain_names
        self.transform = transform
        self.max_sample = max_sample
        self.input_size = input_size
        self._make_dataset()

    def _make_dataset(self):
        if self.test_dir is not None:
            self.imgs = parse_input_folders([self.test_dir], max_sample=self.max_sample)
        logging.info(">> {} samples found in: {}".format(len(self.imgs), self.test_dir))
        return

    def __getitem__(self, index):
        img_path = self.imgs[index]
        try:
            img = Image.open(img_path).convert("RGB")
        except UnidentifiedImageError:
            img = Image.new("RGB", self.input_size)
        # sample = Munch(img=img, fname=[self.imgs[index]])
        sample = Munch(img=img)

        parent, name = str(img_path.parent), img_path.name
        sem_path = os.path.join(
            parent.replace("images", "seg_masks"), name.replace(".jpg", ".png")
        )
        if os.path.exists(sem_path):
            try:
                seg_masks = Image.open(sem_path).convert("L")
            except UnidentifiedImageError:
                seg_masks = Image.new("L", img.size)
        else:
            logging.error(">> No semantic labels found for: {}".format(img_path))
            seg_masks = Image.new("L", img.size)
        sample.seg_masks = seg_masks
        if self.transform is not None:
            sample = self.transform(sample)
        logging.info(">> TestDataset: {sample}")

        return sample

    def __len__(self):
        return len(self.imgs)


def _make_balanced_sampler(labels, target_domain_names=[]):
    logging.info(">> Creating balanced sampler dataset...")
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts
    weights = class_weights[labels]
    return WeightedRandomSampler(weights, len(weights))


def get_train_loader(
    img_size: (int, int) = (256, 256),
    batch_size: int = 8,
    prob: float = 0.5,
    num_workers: int = 8,
    train_list: str = None,
    ref_list=[],
    normalize: str = "imagenet",
    max_scale: float = 2.0,
    max_n_bbox=4,
    target_domain_names=[],
    seg_threshold=0.8,
):
    logging.info("===== Preparing DataLoader =====")

    if normalize == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == "default":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        logging.warning(">> No normalization type specified")
        mean = [1.0, 1.0, 1.0]
        std = [0.0, 0.0, 0.0]

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(1.0, 2.0)),
            transforms.RandomHorizontalFlip(p=prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    transform_custom = transforms.Compose(
        [
            ct.RandomScale(img_size),
            ct.RandomCrop(img_size),
            ct.HorizontalFlip(),
            ct.ToTensor(),
            ct.Normalize(mean=mean, std=std),
            ct.SegMaskToBBoxes([1, 7, 14], n_bbox=max_n_bbox),
            # ct.SegMaskToPatches(8, seg_threshold),
        ]
    )
    dataset = MultiDomainDataset(
        train_list=train_list,
        ref_list=ref_list,
        transform=transform_custom,
        target_domain_names=target_domain_names,
        input_size=img_size,
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,  # use shuffle or sample
        #    sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_ref_loader(
    img_size: (int, int) = (256, 256),
    batch_size: int = 8,
    prob: float = 0.5,
    num_workers: int = 8,
    ref_list: str = None,
    imagenet_normalize: bool = True,
    max_scale: float = 2.0,
    target_domain_names=[],
    max_dataset_size=-1,
    normalize="imagenet",
):
    logging.info("===== Preparing DataLoader =====")

    if normalize == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == "default":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        logging.warning(">> No normalization type specified")
        mean = [1.0, 1.0, 1.0]
        std = [0.0, 0.0, 0.0]

    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=prob),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    dataset = ReferenceDataset(
        ref_list,
        transform,
        target_domain_names=target_domain_names,
        max_sample=max_dataset_size,
        input_size=img_size,
    )

    _domains = [torch.argmax(d) for d in dataset.ref_domains]

    sampler = _make_balanced_sampler(
        torch.Tensor(_domains).to(torch.int), target_domain_names=target_domain_names
    )

    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        #    shuffle=True,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def get_test_loader(
    test_dir="",
    img_size=256,
    batch_size=1,
    normalize="imagenet",
    shuffle=True,
    num_workers=4,
    drop_last=False,
    max_n_bbox=-1,
    seg_threshold=0.8,
    patch_size=8,
):
    logging.info("===== Preparing DataLoader =====")
    if normalize == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == "default":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        logging.warning(">> No normalization type specified")
        mean = [1.0, 1.0, 1.0]
        std = [0.0, 0.0, 0.0]

    transform_custom = transforms.Compose(
        [
            ct.RandomScale(img_size),
            ct.RandomCrop(img_size),
            ct.ToTensor(),
            ct.Normalize(mean=mean, std=std),
            # ct.SegMaskToPatches(patch_size, seg_threshold),
        ]
    )

    dataset = TestDataset(
        test_dir=test_dir, transform=transform_custom, input_size=img_size
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def get_eval_loader(
    test_dir="",
    img_size=256,
    batch_size=1,
    normalize="imagenet",
    shuffle=True,
    num_workers=4,
    drop_last=False,
    max_n_bbox=-1,
    seg_threshold=0.8,
    patch_size=8,
):
    logging.info("Preparing DataLoader for the evaluation phase...")
    if normalize == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
    elif normalize == "default":
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        logging.warning(">> No normalization type specified")
        mean = [1.0, 1.0, 1.0]
        std = [0.0, 0.0, 0.0]

    transform_custom = transforms.Compose(
        [
            ct.RandomScale(img_size),
            ct.RandomCrop(img_size),
            ct.ToTensor(),
            ct.Normalize(mean=mean, std=std),
            # ct.SegMaskToPatches(patch_size, seg_threshold),
        ]
    )

    dataset = TestDataset(
        test_dir=test_dir, transform=transform_custom, input_size=img_size
    )
    return data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


class TrainProvider:
    def __init__(self, loader, latent_dim=16, mode="", num_domains=12):
        self.loader = loader
        self.iter = iter(self.loader)
        self.latent_dim = latent_dim
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
        if x == None:
            self.iter = iter(self.loader)
            x = next(self.iter_ref)
        return x

    def __next__(self):
        # return
        sample = self._fetch_inputs()
        # ref = self._fetch_refs()
        lat_trg = torch.randn(sample.img.size(0), self.latent_dim)
        lat_trg_2 = torch.randn(sample.img.size(0), self.latent_dim)
        inputs = Munch(
            img_src=sample.img,
            d_src=sample.src_domain,
            seg=sample.seg_masks,
            lat_trg=lat_trg,
            lat_trg_2=lat_trg_2,
            bbox=sample.bboxes,
        )
        refs = Munch(img_ref=sample.ref_img, d_trg=sample.trg_domain)
        inputs = Munch({k: v.to(self.device) for k, v in inputs.items()})
        refs = Munch({k: v.to(self.device) for k, v in refs.items()})
        return inputs, refs


class RefProvider:
    def __init__(self, loader_ref):
        self.loader_ref = loader_ref
        self.iter_ref = iter(self.loader_ref)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def _fetch_refs(self, d_src=None):
        try:
            x = next(self.iter_ref, d_src)

        except (AttributeError, StopIteration) as e:
            self.iter_ref = iter(self.loader_ref)
            x = next(self.iter_ref, d_src)
        if x == None:
            self.iter_ref = iter(self.loader_ref)
            x = next(self.iter_ref, d_src)
        return x

    def __next__(self, d_src=None):
        ref = self._fetch_refs()
        if d_src is not None:
            logging.info(
                "d_src:{} ref_domains:{}".format(
                    torch.argmax(d_src), torch.argmax(ref.domain)
                )
            )
            while torch.equal(torch.argmax(d_src), torch.argmax(ref.domain)):
                ref = self._fetch_refs()
        inputs = Munch(img_ref=ref.img, d_trg=ref.domain)
        # TODO make sure not to move to GPU twice
        return Munch({k: v.to(self.device) for k, v in inputs.items()})


class TestProvider:
    def __init__(self, loader_test, mode=""):
        self.loader_test = loader_test
        self.iter_ref = iter(self.loader_test)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.mode = mode

    def _fetch_samples(self, d_src=None):
        try:
            x = next(self.iter_ref, d_src)
        except AttributeError as e:
            self.iter_ref = iter(self.loader_test)
            x = next(self.iter_ref, d_src)
        if x == None:
            self.iter_ref = iter(self.loader_test)
            x = next(self.iter_ref, d_src)
        return x

    def __next__(self):
        sample = self._fetch_samples()
        if self.mode == "val":
            inputs = Munch(img=sample.img, seg=sample.seg_masks)
        elif self.mode == "test":
            inputs = Munch(img=sample.img, seg=sample.seg_masks)
        else:
            raise NotImplementedError
        return Munch(
            {
                k: v.to(self.device) if k is not "fname" else [v]
                for k, v in inputs.items()
            }
        )
