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


def train_model(model, dataloaders, criterions, optimizer, num_epochs, hparams, loss_type='baseline'):
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
            running_vector_loss = 0.0
            # Iterate over data.
            for img_mlo, img_cc, mask_mlo, mask_cc in tqdm(dataloaders[phase]):
                img_mlo, img_cc = img_mlo.cuda(), img_cc.cuda()
                mask_mlo, mask_cc = mask_mlo.cuda(), mask_cc.cuda()

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    pred_mlo, pred_cc = model(img_mlo, img_cc)

                    if loss_type == "baseline":
                        loss = 0.5 * criterions[0](pred_mlo, mask_mlo) + 0.5 * (
                            criterions[0](pred_cc, mask_cc))
                    else:
                        loss = criterions[0](mlo_pred, mask_mlo)

                    dice = 0.5 * dice_metric(torch.sigmoid(pred_mlo), mask_mlo, per_image=True) + 0.5 * dice_metric(
                        torch.sigmoid(pred_cc), mask_cc, per_image=True)

                    vec_mlo = torch.sum(torch.sigmoid(pred_mlo), dim=(1, 2))
                    vec_cc = torch.sum(torch.sigmoid(pred_cc), dim=(1, 2))
                    vector_loss = mse_loss(vec_cc, vec_mlo)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * img_mlo.size(0)
                running_dice += dice * img_mlo.size(0)
                running_vector_loss += vector_loss * img_mlo.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_dice = running_dice / len(dataloaders[phase].dataset)
            epoch_vector_loss = running_vector_loss / len(dataloaders[phase].dataset)

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
                hparams['mse_vector_loss'] = epoch_vector_loss.item()
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


'''

class Train:
    def __init__(self, model, criterion, criterion2, valid_metric, epochs, early_stop, hparams):
        self.loss_fn = criterion
        self.criterion2 = criterion2
        self.model = model
        self.valid_metric = valid_metric
        self.epochs = epochs
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hparams['lr'],
                                          weight_decay=hparams['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min',
                                                                    factor=hparams['scheduler_factor'],
                                                                    patience=hparams['scheduler_patience'], min_lr=3e-7,
                                                                    verbose=True)
        self.early_stop = early_stop
        self.hparams = hparams
        self.epoch = 0

    def train_epoch(self, loader):
        tqdm_loader = tqdm(loader)
        loss_mean = 0
        loss2_mean = 0
        for idx, batch in enumerate(tqdm_loader):
            loss, loss2 = self.batch_train(batch)
            loss_mean = (loss_mean * idx + loss) / (idx + 1)
            loss2_mean = (loss2_mean * idx + loss2) / (idx + 1)
            tqdm_loader.set_description(f'epoch={self.epoch} train: loss = {loss_mean:.4f}, loss2 = {loss2_mean:.3f}')
        return loss_mean

    def batch_train(self, batch):
        img_mlo, img_cc, label_mlo, label_cc = batch
        img_mlo, img_cc, label_mlo, label_cc = img_mlo.cuda(), img_cc.cuda(), label_mlo.cuda(), label_cc.cuda()
        pred_mlo, pred_cc, vec_mlo, vec_cc = self.model(x_mlo=img_mlo, x_cc=img_cc)
        loss = 0.5 * self.loss_fn(pred_mlo, label_mlo) + 0.5 * self.loss_fn(pred_cc, label_cc)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
#        is_require_grad = []
#        for param in self.model.parameters():
#            is_require_grad.append(param.requires_grad)
#        print(is_require_grad)
        return loss, loss

    def valid_epoch(self, loader):
        tqdm_loader = tqdm(loader)
        metric_mean = 0
        loss_mean = 0
        loss2_mean = 0
        for idx, batch in enumerate(tqdm_loader):
            with torch.no_grad():
                loss, loss2, pred_mlo, pred_cc, label_mlo, label_cc = self.batch_valid(batch)
                metric_mlo = self.valid_metric(binarize_mask(pred_mlo, threshold=0.5), label_mlo, per_image=True)
                metric_cc = self.valid_metric(binarize_mask(pred_cc, threshold=0.5), label_cc, per_image=True)
                metric_mean = (metric_mean * idx + (metric_mlo + metric_cc) / 2) / (idx + 1)
                loss_mean = (loss_mean * idx + loss) / (idx + 1)
                loss2_mean = (loss2_mean * idx + loss2) / (idx + 1)
                tqdm_loader.set_description(f'epoch={self.epoch} valid: metric = {metric_mean:.4f}, loss = {loss_mean:.4f}, loss2 = {loss2_mean:.3f} ')
        return loss_mean, metric_mean

    def batch_valid(self, batch):
        img_mlo, img_cc, label_mlo, label_cc = batch
        img_mlo, img_cc, label_mlo, label_cc = img_mlo.cuda(), img_cc.cuda(), label_mlo.cuda(), label_cc.cuda()
        pred_mlo, pred_cc, vec_mlo, vec_cc = self.model(x_mlo=img_mlo, x_cc=img_cc)
        loss = 0.5 * self.loss_fn(pred_mlo, label_mlo) + 0.5 * self.loss_fn(pred_cc, label_cc)
        return loss, loss, pred_mlo, pred_cc, label_mlo, label_cc

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    def load_model(self, best_model_state_dict):
        if type(self.model) == torch.nn.DataParallel:
            self.model.module.load_state_dict(best_model_state_dict)
        else:
            self.model.state_dict(best_model_state_dict)
        print('best model weights are loaded')

    def run_train(self, train_loader, valid_loader):
        best_metric = 0
        best_model_state_dict = None
        for epoch in range(self.epochs):
            self.model.train()
            self.train_epoch(train_loader)
            self.model.eval()
            valid_loss, valid_metric = self.valid_epoch(valid_loader)
            self.epoch += 1
            if best_metric < valid_metric:
                best_metric = valid_metric
                best_model_state_dict = self.get_state_dict(self.model)
            self.scheduler.step(valid_loss)
            self.early_stop(valid_loss, self.model)
            if self.early_stop.early_stop:
                print("Early stopping")
                break
        self.hparams['stopped_epoch'] = self.epoch
        self.load_model(best_model_state_dict)
        self.hparams['model_state_dict'] = best_model_state_dict
        return self.model

'''
