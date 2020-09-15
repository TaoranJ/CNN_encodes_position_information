#!/usr/bin/env python

import os
import argparse

import torch
import numpy as np
from torch import nn
import torch.optim as optimizer
from scipy.stats import pearsonr
from torch.utils.data import DataLoader

from model import PosENet
from data import FMDataset


def init(args):
    """Initial dataloaders, model and optimizer."""

    # Dataset
    train = FMDataset(fms=np.load(args.fm_train), gt_path=args.gt_path,
                      model=args.model)
    test = FMDataset(fms=np.load(args.fm_test), gt_path=args.gt_path,
                     model=args.model)
    synthetic = FMDataset(fms=np.load(args.fm_synthetic), gt_path=args.gt_path,
                          model=args.model)
    input_dim = train[0][0].shape[0]
    train = DataLoader(train, batch_size=args.batch, shuffle=True)
    test = DataLoader(test, batch_size=100000, shuffle=True)
    synthetic = DataLoader(synthetic, batch_size=100000, shuffle=True)
    # Model
    model = PosENet(input_dim=input_dim).to(args.device)
    optim = optimizer.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=1e-4)
    return train, test, synthetic, model, optim


def training(train, model, optim, args):
    """Training procedure"""

    model.train()
    epoch_loss = 0
    for epoch in range(1, 16):
        for batch in train:
            optim.zero_grad()
            fms, gt = batch
            pred = model(fms.to(args.device))
            # Loss calculation (optional flatten operation)
            pred, gt = pred.view(pred.size(0), -1), gt.view(gt.size(0), -1)
            loss = nn.MSELoss()(pred, gt.to(args.device)) / (2 * pred.size(1))
            loss.backward()
            optim.step()
            epoch_loss += loss.detach().item()
        spc, mae = evaluation(train, model, args)
        print('[Epochs: {:02d}/{:02d}] Loss: {:.4f}, '
              'SPC: {:.4f}, MAE: {:.4f}'.format(epoch, 15, epoch_loss,
                                                spc, mae))
        epoch_loss = 0
    return model


def evaluation(data, model, args):
    """Evaluation."""

    model.eval()
    with torch.no_grad():
        preds = torch.tensor([]).to(args.device)
        gts = torch.tensor([]).to(args.device)
        for batch in data:
            fms, gt = batch
            pred = model(fms.to(args.device))
            preds = torch.cat([preds, pred.view(pred.size(0), -1)], dim=0)
            gts = torch.cat([gts, gt.view(gt.size(0), -1).to(args.device)],
                            dim=0)
        # calculate SPC and MAE (optional flatten operation)
        spc = 0
        for ix in range(preds.size(0)):
            spc += pearsonr(preds[ix].to('cpu').numpy(),
                            gts[ix].to('cpu').numpy())[0]
        spc = spc / preds.size(0)
        mae = nn.L1Loss()(preds, gts)
    return spc, mae


def main(args):
    train, test, synthetic, model, optim = init(args)
    model = training(train, model, optim, args)
    spc, mae = evaluation(test, model, args)
    print('[PASCAL-S, {}, {}]: SPC:{:.4f}, MAE:{:.4f}'.format(
        args.model, os.path.basename(args.gt_path), spc, mae))
    spc, mae = evaluation(synthetic, model, args)
    print('[Synthetic data, {}, {}]: SPC:{:.4f}, MAE:{:.4f}'.format(
        args.model, os.path.basename(args.gt_path), spc, mae))


if __name__ == "__main__":
    pparser = argparse.ArgumentParser()
    pparser.add_argument('--model', type=str,
                         choices=['plain', 'vgg', 'resnet'],
                         help='Choice of base architecture.')
    pparser.add_argument('--lr', type=float, default=0.01,
                         help='Learning rate.')
    pparser.add_argument('--batch', type=int, default=4,
                         help='Minibatch size.')
    pparser.add_argument('--gt-path', type=str, help='Path to ground truth.')
    pparser.add_argument('--use-cuda', type=str, default='0',
                         help='Cuda device index.')
    args = pparser.parse_args()
    args.device = 'cuda:' + args.use_cuda if torch.cuda.is_available() \
        else 'cpu'
    args.fm_train = 'fm_train_' + args.model + '.npy'
    args.fm_test = 'fm_test_' + args.model + '.npy'
    args.fm_synthetic = 'fm_synthetic_' + args.model + '.npy'
    main(args)
