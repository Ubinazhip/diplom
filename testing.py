from tqdm import tqdm
import torch
import os
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import copy
import argparse
import albumentations as A
import train
import dataset
import model
from utils.early_stop import EarlyStopping
from utils import losses
from monai.losses import DiceLoss
import inference
import pandas as pd


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--epochs", type=int, default=15, help="number of training epochs")
    parser.add_argument("--fold", type=int, default=0, help="fold number: 0, 1, 2, 3, 4")
    parser.add_argument('--transform', type=str, default=None, help='path to the transform yaml file')
    parser.add_argument('--transform_val', type=str, default=None,
                        help='path to the transform for validation yaml file')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-6, help='weight decay for regularization')
    parser.add_argument('--model', type=str, default='b7', help='model name: unet')
    parser.add_argument('--stop_patience', type=int, default=15, help='patience for early stopping')
    parser.add_argument('--scheduler_factor', type=float, default=0.1, help='reduce factor reduce')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='patience for scheduler')
    parser.add_argument('--loss_weight', type=float, default=1, help='weighted average of losses')
    parser.add_argument('--loss_type', type=str, default='baseline', help='type of the loss')
    parser.add_argument('--save_res_csv', type=str, default=None, help='csv file to save the results')
    parser.add_argument('--dice', type=int, default=1, help='weight for dice loss in ComboLoss')
    parser.add_argument('--bce', type=int, default=0, help='weight for bce loss in ComboLoss')
    parser.add_argument('--focal', type=int, default=0, help='weight for focal loss in ComboLoss')
    parser.add_argument('--dataset', type=str, default='cbis', help='dataset: cbis or inbreast')
    parser.add_argument('--vit', action='store_true')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()
    print(args.vit)
