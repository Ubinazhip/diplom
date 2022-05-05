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
import losses
from monai.losses import DiceLoss


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--epochs", type=int, default=15, help="number of training epochs")
    parser.add_argument("--fold", type=int, default=0, help="fold number: 0, 1, 2, 3, 4")
    parser.add_argument('--transform', type=str, default=None, help='path to the transform yaml file')
    parser.add_argument('--transform_val', type=str, default=None,
                        help='path to the transform for validation yaml file')
    parser.add_argument('--lr', type=float, default=3e-4, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay for regularization')
    parser.add_argument('--file', type=str, default='best_model.pth', help='best model file name')
    parser.add_argument('--model', type=str, default='b7', help='model name: unet')
    parser.add_argument('--stop_patience', type=int, default=15, help='patience for early stopping')
    parser.add_argument('--scheduler_factor', type=float, default=0.1, help='reduce factor reduce')
    parser.add_argument('--scheduler_patience', type=int, default=5, help='patience for scheduler')
    parser.add_argument('--loss_weight', type=float, default=0.5, help='weighted average of losses')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parser()

    model = model.get_model(model_name=args.model)
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    train_loader, valid_loader, test_loader = dataset.data_loader(fold=args.fold, batch_size=args.batch_size,
                                                                  train_transform=args.transform,
                                                                  val_transform=args.transform_val)

    #criterion = losses.ComboLoss({'dice': 1}, channel_weights=[1 / 3, 1 / 3, 1 / 3])
    criterion = DiceLoss(sigmoid=True)
    criterion2 = losses.CustomLoss(loss_weight=args.loss_weight)#torch.nn.MSELoss()
    valid_metric = losses.dice_metric
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f'The model has {pytorch_total_params} parameters, fold = {args.fold}, lr = {args.lr}')
    print(f'Scheduler: patience = {args.scheduler_patience}, reduce factor: {args.scheduler_factor}, loss_weight = {args.loss_weight}')
    hparams = {
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'transform': args.transform,
        'scheduler_factor': args.scheduler_factor,
        'scheduler_patience': args.scheduler_patience,
        'fold': args.fold,
        'loss_weight': args.loss_weight,
        'stop_patience': args.stop_patience,
        'model_name': args.model
    }
    early_stopping = EarlyStopping(patience=args.stop_patience, verbose=True)

    best_metric = train.Train(model=model, criterion=criterion, criterion2=criterion2, valid_metric=valid_metric, epochs=args.epochs,
                              hparams=hparams, early_stop=early_stopping).run_train(train_loader, valid_loader)
