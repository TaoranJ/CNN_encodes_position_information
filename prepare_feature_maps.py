#!/usr/bin/env python

import os
import argparse

import PIL
import torch
import numpy as np
import torchvision
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from torchvision import transforms


# Base architectures
vgg = torchvision.models.vgg16(pretrained=True)
resnet = torchvision.models.resnet152(pretrained=True)
vgg.eval()
resnet.eval()


def fm_resize(fm):
    """Resize feature maps.

    Parameters
    ----------
    fm : :class:`torch.Tensor`
        Feature map, a tensor of shape (3, x, x).

    Returns
    -------
    fm : :class:`torch.Tensor`
        Feature map, a tensor of shape (3, 28, 28).

    """

    return F.interpolate(fm.unsqueeze(0), size=[3, 28, 28], mode="trilinear",
                         align_corners=False).view(3, 28, 28)


def vgg16_fms(img):
    """Get feature maps from pre-trained VGG16. Feature extracted by saving
    model output at the 5th, 10th, 17th, 24th and 31st layer of VGG16.

    Parameters
    ----------
    img : :class:`troch.Tensor`
        Input image, a tensor of shape (3, 224, 224).

    Returns
    -------
    fms : :class:`torch.Tensor`
        5 feature maps, a tensor of shape (5 * 3, 28, 28).

    """

    fms = torch.tensor([]).to(img.device)
    fm_layers = [5, 10, 17, 24, 31]
    modules = list(vgg.children())[0]  # no output dense layers
    for ix in fm_layers:
        sub_model = nn.Sequential(*modules[0:ix]).to(img.device)
        sub_model.eval()
        fms = torch.cat([fms, fm_resize(sub_model(img.unsqueeze(0)))], 0)
    return fms


def resnet152_fms(img):
    """Get feature maps from pre-trained Resnet152. Feature extracted by saving
    model output at the 4th, 5th, 6th, 7th and 8th layer of Resnet152.

    Parameters
    ----------
    image : :class:`troch.Tensor`
        Input image, a tensor of shape (3, 224, 224).

    Returns
    -------
    fms : :class:`torch.Tensor`
        5 feature maps, a tensor of shape (5 * 3, 28, 28).

    """

    fms = torch.tensor([]).to(img.device)
    # conv1, conv2_x, conv3_x, conv4_x, conv5_x
    fm_layers = [4, 5, 6, 7, 8]
    modules = list(resnet.children())
    for ix in fm_layers:
        sub_model = nn.Sequential(*modules[0:ix]).to(img.device)
        sub_model.eval()
        fms = torch.cat([fms, fm_resize(sub_model(img.unsqueeze(0)))], 0)
    return fms


def main(args):
    """Procedure of generating 5 feature maps."""

    # Images are resized to 224 X 224 during training and inference.
    im_resize = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]), ])
    # All feature maps are aligned to a size of 28 X 28.
    fm_align = transforms.Compose([transforms.Resize((28, 28)),
                                   transforms.ToTensor(), ])
    # Generate 5 feature maps for each image
    with torch.no_grad():
        images = list(os.listdir(args.data))
        num_images = len(images)
        for model in ['plain', 'vgg', 'resnet']:
            fms, fm_save_path = torch.tensor([]), 'fm_' + model
            for ix in tqdm(range(num_images), total=num_images, desc=model):
                im_path = os.path.join(args.data, images[ix])
                if model == 'plain':  # simply use the images as features
                    fm = fm_align(PIL.Image.open(im_path))
                else:
                    image = im_resize(PIL.Image.open(im_path))  # 224 X 224
                    if model == 'vgg':
                        fm = vgg16_fms(image.to(args.device))
                    elif model == 'resnet':
                        fm = resnet152_fms(image.to(args.device))
                fms = torch.cat((fms, fm.unsqueeze(0).cpu()), 0)
                np.save(fm_save_path, fms)


if __name__ == "__main__":
    pparser = argparse.ArgumentParser()
    pparser.add_argument('--data', help='Path to dataset.')
    pparser.add_argument('--use-cuda', type=str, default='0',
                         help='Which GPU to use')
    args = pparser.parse_args()
    args.device = 'cuda:' + args.use_cuda if torch.cuda.is_available() \
        else 'cpu'
    main(args)
