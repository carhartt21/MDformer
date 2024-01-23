import sys
from einops import rearrange
import numpy as np
import torch
import random
# from skimage.measure import label, regionprops
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import draw_bounding_boxes
import torch.nn.functional as F
from utils import mask_to_bboxes
import logging


class RandomCrop:
    def __init__(self, size: tuple):
        """
        Initialize RandomCrop transform.

        Args:
            size (tuple): The output size of the crop in the format (height, width).
        """
        assert isinstance(size, tuple), "Size must be a tuple"
        assert len(size) == 2, "Size must have exactly 2 elements"
        
        self.size = size

    def __call__(self, sample):
        """
        Apply random crop transformation to the input sample.

        Args:
            sample: The input sample.

        Returns:
            The transformed sample.
        """
        i, j, h, w = transforms.RandomCrop.get_params(sample.img, output_size=self.size)
        sample.img = transforms.functional.crop(sample.img, i, j, h, w)
        sample.seg_masks = transforms.functional.crop(sample.seg_masks, i, j, h, w)
        _i, _j, _h, _w = transforms.RandomCrop.get_params(sample.ref_img, output_size=self.size)
        sample.ref_img = transforms.functional.crop(sample.ref_img, _i, _j, _h, _w)
        return sample


class RandomScale:
    def __init__(self, min_output_size: tuple):
        """
        Initialize RandomScale transform.

        Args:
            max_scale (float): The maximum scale of the image.
            min_output_size (int): The minimum size of the short side of the scaled image.
        """        
        # assert isinstance(max_scale, float), "Max scale must be a float"
        assert isinstance(min_output_size, tuple), "Minimum output size must be an tuple"
        # self.max_scale = max_scale
        self.output_size = min_output_size

    def __call__(self, sample):
        """
        Apply random scale transformation to the input sample.

        Args:
            sample: The input sample.

        Returns:
            The transformed sample.
        """
        scale, w = self.get_scale(sample.img)
        ref_scale, w_ref = self.get_scale(sample.ref_img)
        sample.img = transforms.functional.resize(sample.img, int(w * scale))
        sample.seg_masks  = transforms.functional.resize(sample.seg_masks , int(w * scale), interpolation=InterpolationMode.NEAREST)
        sample.ref_img = transforms.functional.resize(sample.ref_img, int(w_ref * ref_scale))
        return sample
    
    def get_scale(self, image):
        """
        Get the scale of the input sample.

        Args:
            sample: The input sample.

        Returns:
            The scale of the input sample.
        """
        w, h = image.size[:2]
        min_scale = max(self.output_size[0]/w, self.output_size[1]/h)
        max_scale = min_scale * 1.25
        scale = random.uniform(min_scale, max_scale)
        return scale, w

class HorizontalFlip:
    def __init__(self, p: float = 0.5):
        """
        Initialize HorizontalFlip transform.

        Args:
            p (float): The probability of applying horizontal flip. Default is 0.5.
        """
        assert isinstance(p, float), "Probability must be a float"
        assert 0 <= p <= 1, "Probability must be between 0 and 1"        
        self.p = p

    def __call__(self, sample):
        """
        Apply horizontal flip transformation to the input sample.

        Args:
            sample: The input sample.

        Returns:
            The transformed sample.
        """
        image = sample.img
        seg_masks = sample['seg_masks']
        if random.random() < self.p:
            sample.img = transforms.functional.hflip(image)
            sample.seg_masks = transforms.functional.hflip(seg_masks)
            sample.ref_img = transforms.functional.hflip(sample.ref_img)
        return sample

class Normalize:
    def __init__(self, mean: list, std: list):
        """
        Initialize Normalize transform.

        Args:
            mean (list): The mean values for normalization.
            std (list): The standard deviation values for normalization.
        """
        assert isinstance(mean, list), "Mean must be a list"
        assert isinstance(std, list), "Std must be a list"
        assert len(mean) == 3, "Mean must have exactly 3 elements"
        assert len(std) == 3, "Std must have exactly 3 elements"
        
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        """
        Apply normalization transformation to the input sample.

        Args:
            sample: The input sample.

        Returns:
            The transformed sample.
        """
        sample.img = transforms.functional.normalize(sample.img, self.mean, self.std)
        sample.ref_img = transforms.functional.normalize(sample.ref_img, self.mean, self.std)
        return sample

class ToTensor:
    def __call__(self, sample):
        """
        Convert the input sample to tensor.

        Args:
            sample: The input sample.

        Returns:
            The transformed sample.
        """
        sample.img = transforms.functional.to_tensor(sample.img)
        sample.ref_img = transforms.functional.to_tensor(sample.ref_img)
        sample.seg_masks = transforms.functional.pil_to_tensor(sample.seg_masks).to(torch.int32)
        # sample.bboxes = torch.tensor(sample.bboxes)
        return sample

class SegMaskToPatches:
    def __init__(self, patch_size: int = 16, min_coverage: float = 0.9):
        """
        Initialize SegMaskToPatches transform.

        Args:
            patch_size (int): The size of the patches.
            min_coverage (float): The minimum percentage of pixels that should be covered by a class in a patch for the class to be assigned to the patch.
        """
        assert isinstance(patch_size, int), "Patch size must be an int"
        assert isinstance(min_coverage, float), "Minimum coverage must be a float"
        assert 0 <= min_coverage <= 1, "Minimum coverage must be between 0 and 1"
        
        self.patch_size = patch_size
        self.min_coverage = min_coverage

    def __call__(self, sample):
        """
        Convert the segmentation map in the input sample to tensor.

        Args:
            sample: The input sample.

        Returns:
            The transformed sample.
        """
        height, width = sample.seg_masks.shape[-2]//4, sample.seg_masks.shape[-1]//4
        assert height == width, "Segmentation map height {} and width {} are not equal".format(height, width)
        assert height % self.patch_size == 0, "Segmenation map {}x{} is not divisible by patch size {}".format(height, width, self.patch_size)

        n_patches = (height // self.patch_size)
        # Get list of unique classes in the segmentation map excluding 0
        _classes = torch.unique(sample.seg_masks[sample.seg_masks != 0], sorted=True)
        # Calculate the minimum number of pixels that should be covered by the segmentation map in each patch
        min_pixels = self.patch_size ** 2 * self.min_coverage
        # Create a grid to hold the classes assigned to each patch
        grid = torch.zeros((n_patches, n_patches), dtype=torch.int32)

        # Iterate over each cell in the grid
        for i in range(n_patches):
            for j in range(n_patches):
                top = i * self.patch_size
                bottom = (i + 1) * self.patch_size
                left = j * self.patch_size
                right = (j + 1) * self.patch_size                
                # Iterate over each class in the segmentation map
                for c in _classes:
                    # Count the number of pixels covered by the current class in the current cell
                    pixels_covered = torch.sum(sample.seg_masks[:, top:bottom, left:right] == c)
                    
                    # If the number of pixels covered is greater than or equal to the minimum required, assign the class to the current cell
                    if pixels_covered >= min_pixels:
                        grid[i, j] = c

        sample.seg_masks= grid.view(-1).to(torch.int32)
        return sample


class SegMaskToBBoxes:
    def __init__(self, fg_classes: list=[14], min_bbox_size: int=512, min_pixel = 256, min_extent: float=0.5, n_bbox=8):
        """
        Initialize SegMaskToBBoxes transform.
        fg_classes (list): The list of classes to consider as foreground.
        min_size (int): The minimum size of the bounding box.
        min_extent (float): The minimum extent of the bounding box.
        max_n_bbox (int): The maximum number of bounding boxes to return.
        """
        self.fg_classes = fg_classes
        self.min_bbox_size = min_bbox_size
        self.min_extent = min_extent
        self.n_bbox = n_bbox    
        self.min_pixel = min_pixel

    def __call__(self, sample):
        """
        Convert the segmentation map in the input sample to tensor.
        Args:
            sample: The input sample.  
        Returns:
            The transformed sample.
        """
        boxes = torch.zeros((self.n_bbox, 5))        
        fg_classes = torch.tensor(self.fg_classes)
        _classes = torch.unique(sample.seg_masks[sample.seg_masks != 0], sorted=True)
        i = 0        
        for c in fg_classes:
            _boxes = []
            if c in _classes:
                mask  = sample.seg_masks == c
                if mask.sum() < self.min_pixel:
                    continue

                _boxes = mask_to_bboxes(mask.squeeze(), min_size=self.min_pixel, pos_thres=0.0)
                for box in _boxes:
                    if (box[2] - box[0]) * (box[3] - box[1]) < self.min_bbox_size: 
                        continue
                    box = box/sample.seg_masks[0].shape[0]
                    boxes[i] = torch.cat([torch.Tensor([c]).to(box.dtype), box])
                    i += 1
                    if i == self.n_bbox:
                        sample['bboxes'] = boxes
                        return sample
        sample['bboxes'] = boxes
        return sample

    
class UnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor    
    

