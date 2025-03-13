import os
import torch
import random
import torchvision.transforms.functional as F

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop

class_to_name = {
  0: 'unlabelled',
  1: 'building',
  2: 'fence',
  3: 'other',
  4: 'pedestrian',
  5: 'pole',
  6: 'road line',
  7: 'road',
  8: 'sidewalk',
  9: 'vegetation',
  10: 'car',
  11: 'wall',
  12: 'traffic sign'
}

class RandomCropSync:
  """
    A class that performs a synchronized random crop on an image and its
    corresponding mask.

    Attributes:
      size (tuple[int, int]): The target crop size (h,w).
  """
  def __init__(self, size):
    self.size = size

  def __call__(self, img, mask):
    i, j, h, w = RandomCrop.get_params(img, output_size=self.size)

    img = F.crop(img, i, j, h, w)
    mask = F.crop(mask, i, j, h, w)
    
    return img, mask


class RandomHorizontalFlipSync:
  """
    A class that randomly flips an image and its corresponding mask horizontally
    with a given probability.

    Attributes:
      p (float): The probability of applying the horizontal flip.
  """
  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, img, mask):
    if random.random() < self.p:
      img = F.hflip(img)
      mask = F.hflip(mask)
    return img, mask


class RandomVerticalFlipSync:
  """
    A class that randomly flips an image and its corresponding mask vertically
    with a given probability.

    Attributes:
      p (float): The probability of applying the vertical flip.
  """
  def __init__(self, p=0.5):
    self.p = p

  def __call__(self, img, mask):
    if random.random() < self.p:
      img = F.vflip(img)
      mask = F.vflip(mask)
    return img, mask
    

class ToTensorSync:
  """
    A class that converts an image and its corresponding mask into PyTorch
    tensors.
  """
  def __call__(self, img, mask):
    img_tensor = F.to_tensor(img)
    mask_tensor = F.pil_to_tensor(mask)
    return img_tensor, mask_tensor
    

class NormalizeSync:
  """
    A class that normalizes a tensor image using provided mean and standard
    deviation values while leaving the mask unchanged.

    Attributes:
      mean (list[float]): The mean values for normalization.
      std (list[float]): The standard deviation values for normalization.
  """
  def __init__(self, mean, std):
    self.mean = mean
    self.std = std

  def __call__(self, img, mask):
    img = F.normalize(img, mean=self.mean, std=self.std)
    return img, mask
        

class ComposeSync:
  """
    A class that composes multiple synchronous transformation functions and
    applies them sequentially to an image-mask pair.

    Attributes:
      transforms (list[callable]): A list of transformation to be applied.
  """
  def __init__(self, transforms):
    self.transforms = transforms

  def __call__(self, img, mask):
    for transform in self.transforms:
        img, mask = transform(img, mask)
    return img, mask


class SegmentationDataset(Dataset):
  """
    A Dataset for segmentation tasks that loads images and corresponding masks
    from specified directories.

    This class allows files to be stored in separate directories.
    File names are assumed to be in the same order.

    Attributes:
      images_dirs (str or list[str]): List of directories containing image files.
      masks_dirs (str or list[str]): List of directories containing mask files.
      transforms (callable, optional): A function to apply transformations to the image and mask.
  """
  def __init__(self, images_dirs, masks_dirs, transforms=None):

    if isinstance(images_dirs, str):
      images_dirs = [images_dirs]
    if isinstance(masks_dirs, str):
      masks_dirs = [masks_dirs]

    self.images_dirs = images_dirs
    self.masks_dirs = masks_dirs
    self.transforms = transforms

    self.image_paths = []
    for img_dir in self.images_dirs:
      files = sorted(os.listdir(img_dir))
      self.image_paths.extend([os.path.join(img_dir, f) for f in files])

    self.mask_paths = []
    for mask_dir in self.masks_dirs:
      files = sorted(os.listdir(mask_dir))
      self.mask_paths.extend([os.path.join(mask_dir, f) for f in files])

    if len(self.image_paths) != len(self.mask_paths):
      raise ValueError("The number of images and masks must be equal.")

  def __len__(self):
    return len(self.image_paths)

  def __getitem__(self, idx):
    img_path = self.image_paths[idx]
    mask_path = self.mask_paths[idx]

    image = Image.open(img_path).convert("RGB")
    mask = Image.open(mask_path).convert("L")

    if self.transforms:
      image, mask = self.transforms(image, mask)

    mask = mask.squeeze()
    mask = mask.type(torch.long) - 1

    return image, mask