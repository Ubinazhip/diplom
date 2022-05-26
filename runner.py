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


def save_res_csv(res_dict, file):
    if not os.path.exists(file):
        df = pd.DataFrame(data=res_dict, index=[0])
        df.to_csv(file, index=False)
    else:
        df = pd.read_csv(file)
        df = df.append(res_dict, ignore_index=True)
        df.to_csv(file, index=False)


if __name__ == '__main__':
    args = parser()
    model_splits = args.model.split('_')
    model_name = model_splits[0]
    if len(model_splits) >= 2:
        encoder = args.model.split('_')[1]
    else:
        encoder = None
    model = model.get_model(model_name=model_name, encoder=encoder)
    model = model.cuda()
    model = torch.nn.DataParallel(model)

    dataloaders = dataset.data_loader(fold=args.fold, batch_size=args.batch_size,
                                      train_transform=args.transform, val_transform=args.transform_val,
                                      dataset=args.dataset, vit=args.vit)

    criterion = losses.ComboLoss(dict(dice=args.dice, bce=args.bce, focal=args.focal))
    criterion2 = losses.CustomLoss(loss_weight=args.loss_weight)  # torch.nn.MSELoss()
    criterions = [criterion, criterion2]
    valid_metric = losses.dice_metric
    pytorch_total_params = sum(p.numel() for p in model.parameters())

    print('-' * 100)
    print(f'The model has {pytorch_total_params} parameters, fold = {args.fold}, lr = {args.lr},'
          f'epochs = {args.epochs}, dataset = {args.dataset}, vision transformers: {args.vit}')
    print(
        f'Scheduler: patience = {args.scheduler_patience}, reduce factor: {args.scheduler_factor}, loss_weight = {args.loss_weight}')
    print(f"ComboLoss: dice = {args.dice}, bce = {args.bce}, focal = {args.focal}; loss_type = {args.loss_type}, "
           f"lr = {args.lr}")
    print('-' * 100)
    print('\n')

    hparams = dict(model_name=args.model, fold=args.fold, dice_bce_focal=f'{args.dice}, {args.bce}, {args.focal}',
                   train_metric=0, val_metric=0, test_metric=0, train_loss=0, val_loss=0, test_loss=0,
                   stopped_epoch=0, mse_vector_loss=0.0, transform=args.transform, weight_decay=args.weight_decay,
                   stop_patience=args.stop_patience, scheduler_patience=args.scheduler_patience,
                   scheduler_factor=args.scheduler_factor, lr=args.lr, loss_weight=args.loss_weight,
                   loss_type=args.loss_type, dataset=args.dataset, model_state_dict=None)

    early_stopping = EarlyStopping(patience=args.stop_patience, verbose=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                 weight_decay=args.weight_decay)

    model = train.train_model(model=model, dataloaders=dataloaders, criterions=criterions, optimizer=optimizer,
                              num_epochs=args.epochs, hparams=hparams, loss_type=args.loss_type, vit=args.vit)

    dataloaders = dataset.data_loader(fold=args.fold, batch_size=1,
                                      train_transform=args.transform_val, val_transform=args.transform_val,
                                      dataset=args.dataset, vit=args.vit)

    model.eval()
    test_metric, test_loss = inference.inference(model, dataloaders['test'], criterion=criterion,
                                                 metric=valid_metric, mode='test', vit=args.vit)
    val_metric, val_loss = inference.inference(model, dataloaders['val'], criterion=criterion,
                                               metric=valid_metric, mode='validation', vit=args.vit)
    train_metric, train_loss = inference.inference(model, dataloaders['train'], criterion=criterion,
                                                   metric=valid_metric, mode='train', vit=args.vit)

    hparams['val_metric'], hparams['val_loss'] = val_metric, val_loss
    hparams['test_metric'], hparams['test_loss'] = test_metric, test_loss
    hparams['train_metric'], hparams['train_loss'] = train_metric, train_loss
    if args.dataset == 'inbreast':
        save_model = test_metric > 0.7 and val_metric > 0.7 and train_metric > 0.7
    else:
        save_model = test_metric > 0.55 and val_metric > 0.55 and train_metric > 0.55

    if save_model:
        name_file = f'./best_models/{args.model}_{args.dataset}_fold{args.fold}_dice_train{train_metric:.3f}_val{val_metric:.3f}_test{test_metric:.3f}_stop{args.stop_patience}{args.scheduler_patience}.pth'
        torch.save(hparams, name_file)
        print(f'The best model is saved as {name_file}')
    else:
        print('The best model will not be saved, because it has too low dice')

    hparams.pop('model_state_dict', None)
    if args.save_res_csv:
        save_res_csv(res_dict=hparams, file=args.save_res_csv)
