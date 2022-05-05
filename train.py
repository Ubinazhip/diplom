import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import losses
from tqdm import tqdm


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

    def batch_train(self, batch):
        img_mlo, img_cc, label_mlo, label_cc = batch
        img_mlo, img_cc, label_mlo, label_cc = img_mlo.cuda(), img_cc.cuda(), label_mlo.cuda(), label_cc.cuda()
        pred_label_mlo, pred_label_cc, vec_mlo, vec_cc = self.model(x_mlo=img_mlo, x_cc=img_cc)
        #loss2 = self.criterion2(output_mlo=vec_mlo, output_cc=vec_cc)
        loss = self.criterion2(pred_mlo=pred_label_mlo, pred_cc=pred_label_cc, true_mlo=label_mlo, true_cc=label_cc,
                                output_mlo=vec_mlo, output_cc=vec_cc)
        self.optimizer.zero_grad()
        loss2 = 0.5 * self.loss_fn(pred_label_mlo, label_mlo) + 0.5 * self.loss_fn(pred_label_cc, label_cc)
        loss.backward()
        self.optimizer.step()
        return loss, loss2

    def valid_epoch(self, loader):
        tqdm_loader = tqdm(loader)
        metric_mean = 0
        loss_mean = 0
        loss2_mean = 0
        for idx, batch in enumerate(tqdm_loader):
            with torch.no_grad():
                loss, loss2, pred_label_mlo, pred_label_cc, label_mlo, label_cc = self.batch_valid(batch)
                metric_mlo = self.valid_metric(pred_label_mlo, label_mlo)
                metric_cc = self.valid_metric(pred_label_cc, label_cc)
                metric_mean = (metric_mean * idx + (metric_mlo + metric_cc) / 2) / (idx + 1)
                loss_mean = (loss_mean * idx + loss) / (idx + 1)
                loss2_mean = (loss2_mean * idx + loss2) / (idx + 1)
                tqdm_loader.set_description(f'epoch={self.epoch} valid: metric = {metric_mean:.4f}, loss = {loss_mean:.4f}, loss2 = {loss2_mean:.3f} ')
                                           # f'mlo_metric = {metric_mlo:.4f}, cc metric={metric_cc:.4f}')
        return loss_mean, metric_mean

    def batch_valid(self, batch):
        img_mlo, img_cc, label_mlo, label_cc = batch
        img_mlo, img_cc, label_mlo, label_cc = img_mlo.cuda(), img_cc.cuda(), label_mlo.cuda(), label_cc.cuda()
        pred_label_mlo, pred_label_cc, vec_mlo, vec_cc = self.model(x_mlo=img_mlo, x_cc=img_cc)
        loss = self.criterion2(pred_mlo=pred_label_mlo, pred_cc=pred_label_cc, true_mlo=label_mlo, true_cc=label_cc, output_mlo=vec_mlo, output_cc=vec_cc)
        loss2 = 0.5 * self.loss_fn(pred_label_mlo, label_mlo) + 0.5 * self.loss_fn(pred_label_cc, label_cc)
        return loss, loss2, torch.sigmoid(pred_label_mlo), torch.sigmoid(pred_label_cc), label_mlo, label_cc

    @staticmethod
    def get_state_dict(model):
        if type(model) == torch.nn.DataParallel:
            state_dict = model.module.state_dict()
        else:
            state_dict = model.state_dict()
        return state_dict

    @staticmethod
    def binarize_mask(mask, threshold=0.5):
        mask_sigm = torch.sigmoid(mask)
        mask_binar = torch.where(mask_sigm > threshold, torch.ones_like(mask_sigm), torch.zeros_like(mask_sigm))
        return mask_binar

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
        if best_metric > 0.2:
            name_file = f'./best_models/{self.hparams["model_name"]}_fold{self.hparams["fold"]}_metric{best_metric:.3f}_weight{self.hparams["loss_weight"]}_stop{self.hparams["stop_patience"]}{self.hparams["scheduler_patience"]}.pth'
            self.hparams['model_state_dict'] = best_model_state_dict
            torch.save(self.hparams, name_file)
            print(f'The best model is saved as {name_file}')
        else:
            print(f'The model will not be saved, because best metric is {best_metric}...')
        return best_metric
