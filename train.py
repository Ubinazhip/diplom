import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
import copy
import time
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, MultiStepLR, CyclicLR
import sys
import os
from utils import losses
from utils.early_stop import EarlyStopping
from utils.dct_encoder import DctMaskEncoding


def get_state_dict(model):
    if type(model) == torch.nn.DataParallel:
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    return state_dict


def load_model(model, best_model_state_dict):
    if type(model) == torch.nn.DataParallel:
        model.module.load_state_dict(best_model_state_dict)
    else:
        model.state_dict(best_model_state_dict)
    print('best model weights are loaded ...')
    print()
    return model


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def train_model(model, dataloaders, criterions, optimizer, num_epochs, hparams, loss_type='baseline', vit=False):
    since = time.time()

    val_dice_history = []

    best_dice = 0.0

    dice_metric = losses.dice_metric
    #    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.3)
    early_stopping = EarlyStopping(patience=hparams['stop_patience'], verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=hparams['scheduler_factor'],
                                                           patience=hparams['scheduler_patience'], min_lr=3e-8,
                                                           verbose=True)
    mse_loss = torch.nn.MSELoss()
    jensen_shannon = losses.JSD()
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_dice = 0.0
            running_vector_loss = 0.0 #mse on y axis
            running_vector_loss_x = 0.0 #mse_on x axis
            running_vector_jensen_x = 0.0 #jensen shannon on x axis
            # Iterate over data.
            for img_mlo, img_cc, mask_mlo, mask_cc in tqdm(dataloaders[phase]):
                if vit:
                    img_mlo, mask_mlo = img_mlo.cuda(), mask_mlo.cuda()
                else:
                    img_mlo, img_cc = img_mlo.cuda(), img_cc.cuda()
                    mask_mlo, mask_cc = mask_mlo.cuda(), mask_cc.cuda()

                optimizer.zero_grad()
                mask_encoder = DctMaskEncoding(vec_dim=5000, mask_size=mask_cc.shape[-1])
                with torch.set_grad_enabled(phase == 'train'):
                    if vit:
                        pred_mlo = model(img_mlo)
                        loss = criterions[0](pred_mlo, mask_mlo)
                        #dice = dice_metric(torch.sigmoid(pred_mlo), mask_mlo, per_image=True)
                        splits_img, splits_mask = torch.split(pred_mlo, 512, dim=2), torch.split(mask_mlo, 512, dim=2)
                        pred_mlo, pred_cc = splits_img[0], splits_img[1]
                        mask_mlo, mask_cc = splits_mask[0], splits_mask[1]
                        dice = 0.5 * dice_metric(torch.sigmoid(pred_mlo), mask_mlo, per_image=True) + 0.5 * dice_metric(
                            torch.sigmoid(pred_cc), mask_cc, per_image=True)

                    else:
                        pred_mlo, pred_cc = model(img_mlo, img_cc)
                        loss = 0.5 * criterions[0](pred_mlo, mask_mlo) + 0.5 * (
                            criterions[0](pred_cc, mask_cc))
                        dice = 0.5 * dice_metric(torch.sigmoid(pred_mlo), mask_mlo, per_image=True) + 0.5 * dice_metric(
                            torch.sigmoid(pred_cc), mask_cc, per_image=True)

                    vec_mlo = torch.sum(torch.sigmoid(pred_mlo), dim=(1, 2))
                    vec_cc = torch.sum(torch.sigmoid(pred_cc), dim=(1, 2))
                    vector_loss = mse_loss(vec_cc, vec_mlo)

                    vec_mlo_x = torch.sum(torch.sigmoid(pred_mlo), dim=(1, 3))
                    vec_cc_x = torch.sum(torch.sigmoid(pred_cc), dim=(1, 3))
                    vector_loss_x = mse_loss(vec_cc_x, vec_mlo_x)
                    vector_jensen_x = jensen_shannon(vec_cc_x, vec_mlo_x)

                    if loss_type == 'axis_sum':
                        loss = hparams['loss_weight'] * loss + (1 - hparams['loss_weight']) * vector_loss
                    elif loss_type == 'dct_encoder':
                        encoded_mlo = mask_encoder.encode(torch.sigmoid(pred_mlo).detach().cpu())
                        encoded_cc = mask_encoder.encode(torch.sigmoid(pred_cc).detach().cpu())
                        encoded_mlo = encoded_mlo.cuda()
                        encoded_cc = encoded_cc.cuda()
                        dct_loss = mse_loss(encoded_mlo, encoded_cc)
                        loss = hparams['loss_weight'] * loss + (1 - hparams['loss_weight']) * dct_loss

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * img_mlo.size(0)
                running_dice += dice.item() * img_mlo.size(0)
                running_vector_loss += vector_loss.item() * img_mlo.size(0)
                running_vector_loss_x += vector_loss_x.item() * img_mlo.size(0)
                running_vector_jensen_x += vector_jensen_x.item() * img_mlo.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_dice = running_dice / len(dataloaders[phase].dataset)
            epoch_vector_loss = running_vector_loss / len(dataloaders[phase].dataset)
            epoch_vector_loss_x = running_vector_loss_x / len(dataloaders[phase].dataset)
            epoch_vector_jensen_x = running_vector_jensen_x / len(dataloaders[phase].dataset)

            if phase == "val":
                scheduler.step(epoch_loss)
                print()

            print('{} Loss: {:.4f} Dice: {:.4f} Vector loss: {:.4f}'.format(phase, epoch_loss, epoch_dice, epoch_vector_loss))

            if epoch % 5 == 0:
                print('learning_rate:', str(get_lr(optimizer)))
            # deep copy the model
            if phase == 'val' and epoch_dice > best_dice:
                best_dice = epoch_dice
                best_acc_model_wts = copy.deepcopy(get_state_dict(model))
                hparams['model_state_dict'] = best_acc_model_wts
                hparams['mse_vector_loss'] = epoch_vector_loss
                hparams['mse_vector_loss_x'] = epoch_vector_loss_x
                hparams['jensen_vector_x'] = epoch_vector_jensen_x

            if phase == 'val':
                val_dice_history.append(epoch_dice)
                early_stopping(epoch_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            hparams['stopped_epoch'] = epoch
            break

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val dice: {:4f}, mse vector loss: {:.4f}'.format(best_dice, hparams['mse_vector_loss']))
    model = load_model(model, best_acc_model_wts)
    return model


