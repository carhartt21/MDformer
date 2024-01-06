import torch
import random
import torchvision.transforms as transforms

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
    def __init__(self, max_scale: float, min_output_size: int):
        """
        Initialize RandomScale transform.

        Args:
            max_scale (float): The maximum scale of the image.
            min_output_size (int): The minimum size of the short side of the scaled image.
        """        
        assert isinstance(max_scale, float), "Max scale must be a float"
        assert isinstance(min_output_size, int), "Minimum output size must be an int"
        self.max_scale = max_scale
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
        image_size = image.size
        min_scale = max(self.output_size/image_size[0], self.output_size/image_size[1])
        if min_scale > self.max_scale:
            scale = min_scale
        else:     
            scale = random.uniform(min_scale, self.max_scale)
        h, w = image.size[:2]
        new_size = (int(h * scale), int(w * scale))
        if isinstance(new_size, int):
            if h > w:
                new_h, new_w = new_size * h / w, new_size
            else:
                new_h, new_w = new_size, new_size * w / h
        else:
            new_h, new_w = new_size

        new_h, new_w = int(new_h), int(new_w)

        sample.img = transforms.functional.resize(image, (new_h, new_w))
        sample.seg_mask  = transforms.functional.resize(sample.seg_mask , (new_h, new_w))

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
        sample.seg_mask = transforms.functional.to_tensor(sample.seg_mask)
        return sample
    
    
