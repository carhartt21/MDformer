import sys
import numpy as np
import torch
import random
from skimage.measure import label, regionprops
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode


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
        sample['seg_mask'] = transforms.functional.crop(sample['seg_mask'], i, j, h, w)
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
        image = sample.img
        w, h = image.size[:2]
        min_scale = max(self.output_size[0]/w, self.output_size[1]/h)
        max_scale = min_scale * 1.25
        scale = random.uniform(min_scale, max_scale)
        # new_size = (int(w * scale), int(h * scale))
        # print("image.size: {}, scale: {}, new_size: {}, min_scale: {}, max_scale: {}".format(image.size, scale, new_size, min_scale, max_scale))
        # new_h, new_w = int(new_h), int(new_w)
        sample.img.save('before_resize.jpg')
        sample.img = transforms.functional.resize(image, int(w * scale))
        sample.seg_mask  = transforms.functional.resize(sample.seg_mask , int(w * scale), interpolation=InterpolationMode.NEAREST)
        sample.img.save('after_resize.jpg')
        return sample
    
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
        seg_mask = sample['seg_mask']
        if random.random() < self.p:
            sample.img = transforms.functional.hflip(image)
            sample.seg_mask = transforms.functional.hflip(seg_mask)
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
        sample.seg_mask = transforms.functional.pil_to_tensor(sample.seg_mask)
        sample.bboxes = torch.tensor(sample.bboxes)
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
      
        height, width = sample.seg_mask.shape[-2:]
        assert height == width, "Segmenation map height {} and width {} are not equal".format(height, width)
        assert height % self.patch_size == 0, "Segmenation map {}x{} is not divisible by patch size {}".format(height, width, self.patch_size)

        n_patches = (height // self.patch_size)
        # Get list of unique classes in the segmentation map
        _classes = torch.unique(sample.seg_mask, sorted=True)
        # Calculate the minimum number of pixels that should be covered by the segmentation map in each patch
        min_pixels = self.patch_size ** 2 * self.min_coverage
        # Create a grid to hold the classes assigned to each patch
        grid = torch.full((n_patches, n_patches), -1, dtype=torch.int16)

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
                    pixels_covered = torch.sum(sample.seg_mask[:, top:bottom, left:right] == c)
                    
                    # If the number of pixels covered is greater than or equal to the minimum required, assign the class to the current cell
                    if pixels_covered >= min_pixels:
                        grid[i, j] = c

        sample.seg_mask= grid.view(-1)
        return sample


class SegMaskToBBoxes:
    def __init__(self, fg_classes: list=[1, 7, 14], min_size: int=256, min_extent: float=0.5, max_n_bbox=8):
        """
        Initialize SegMaskToBBoxes transform.
        fg_classes (list): The list of classes to consider as foreground.
        min_size (int): The minimum size of the bounding box.
        min_extent (float): The minimum extent of the bounding box.
        max_n_bbox (int): The maximum number of bounding boxes to return.
        """
        self.fg_classes = fg_classes
        self.min_size = min_size
        self.min_extent = min_extent
        self.max_n_bbox = max_n_bbox    

    def __call__(self, sample):
        """
        Convert the segmentation map in the input sample to tensor.
        Args:
            sample: The input sample.  
        Returns:
            The transformed sample.
        """

        seg_mask = np.asarray(sample.seg_mask)
        bboxes = []
        img = sample.img
        regions = []
        sample.img.save('img.png')
        for class_id in self.fg_classes:
            class_pixels = seg_mask == class_id
            if class_pixels.sum() > 0: 
                labels = label(class_pixels)
                regions += regionprops(labels)
        # sort by area
        while (len(bboxes) < self.max_n_bbox):
            for prop in sorted(
                regions,
                key=lambda r: r.area,
                reverse=True,
            ):
                if prop.area > self.min_size and prop.extent > self.min_extent:
                    bboxes.append(prop.bbox)
                else:
                    continue
        sample['bboxes'] = bboxes
        return sample
    

