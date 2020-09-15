# CNN Encodes Position Information
Implementation of an ICLR 2020 paper: 'How Much Position Information Do Convolutional Neural Networks Encode?'

# Quickstart

## Step 0: download datsets

`DUT-S` training dataset is here: http://saliencydetection.net/duts/download/DUTS-TR.zip. `DUTS-TR-Image` is used for training.

`DUT-S` test dataset is here: http://saliencydetection.net/duts/download/DUTS-TE.zip. The paper didn't use it.

`PASCAL-S` dataset is here: http://cbs.ic.gatech.edu/salobj/. The paper used all images in `/datasets/imgs/pascal/`. Delete `imgList.mat` and `Thumbs.db` before using the folder.

## Step 1: prepare feature maps

```bash
# DUT-S training used as training set
python prepare_feature_maps.py --data path_to_DUTS-TR-Image --use-cuda 0 --label train
# PASCAL-S used as test set
python prepare_feature_maps.py --data path_to_pascal --use-cuda 0 --label test
# Synthetic pictures used as test set
python prepare_feature_maps.py --data synthetic_data --use-cuda 0 --label synthetic
```

## Step 2: train and evaluation the model

```bash
# Specify the architecture, the ground truth, the learning rate and the minibatch size
python learn.py --model resnet --gt-path ground_truth/gt_gau.png --lr 0.001 --batch 4
```
