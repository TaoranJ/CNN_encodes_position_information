#!/usr/bin/env python


import PIL
import torch
from torchvision import transforms
from torch.utils.data import Dataset


class FMDataset(Dataset):
    """Read featuremaps and ground truth.

    Parameters
    ----------
    fms : :class:`numpy.ndarray`
        Feature maps for each image.
    gt_path : str
        Path to ground truth image.
    model : str
        Are feature maps from plain, VGG16, or Resnet152?

    """

    def __init__(self, fms, gt_path, model):
        self.fms = fms
        self.gt_path = gt_path
        self.model = model
        # Ground truth
        gt = PIL.Image.open(self.gt_path)
        self.gt = transforms.Compose(
                [transforms.Resize((26, 26)), transforms.ToTensor(), ])(gt)

    def __len__(self):
        return self.fms.shape[0]

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError('{} index out of range'.format(
                self.__class__.__name__))
        # Feature maps
        if self.model == 'plain':
            fm = torch.tensor(self.fms[index] * 255)
        else:
            fm = torch.tensor(self.fms[index])
        return fm, self.gt
