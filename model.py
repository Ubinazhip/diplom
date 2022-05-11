import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from monai.networks.nets import UNet,UNETR
from models import ternausnets
from models.selim_zoo.unet import Resnet
import segmentation_models_pytorch as smp


def get_model(model_name='unet'):
    if model_name == 'unet':
        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(4, 8, 16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2, 2, 2),
        )
        model = Model(model)#MyModel(model, img_size=512)
    elif model_name == 'unet_resnet50':
        model = smp.Unet(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        model = Model(model)
    elif model_name == 'unet_resnet34':
        model = smp.Unet(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        model = Model(model)
    elif model_name == 'unetpp_resnet34':
        model = smp.UnetPlusPlus(
            encoder_name="resnet34",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        model = Model(model)
    elif model_name == 'unetpp_resnet50':
        model = smp.UnetPlusPlus(
            encoder_name="resnet50",
            encoder_weights="imagenet",
            in_channels=3,
            classes=1,
        )
        model = Model(model)
    else:
        raise NotImplementedError('No such model in our zoo...')
    print(f'The model {model_name} is loaded')
    return model


class Model(torch.nn.Module):

    def __init__(self, model):
        super(Model, self).__init__()
        self.model = model

    def forward_once(self, x):
        pred = self.model(x)
        return pred

    def forward(self, x_mlo, x_cc):
        pred_mlo = self.forward_once(x_mlo)
        pred_cc = self.forward_once(x_cc)
        return pred_mlo, pred_cc

    
class MyModel(torch.nn.Module):  # info about img + info about mask

    def __init__(self, model, vec_length=256, img_size=1024, weight_mask=0.8):
        super(MyModel, self).__init__()
        self.model = model
        self.weight_mask = weight_mask
        self.in_features = 675 if img_size == 1024 else 147
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2),

            nn.Conv2d(8, 16, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(16, 6, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(6, 3, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),

            nn.Flatten()
        )

        self.linear_block = nn.Sequential(
            nn.Linear(self.in_features, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, vec_length),

        )

    def forward_once(self, x):
        pred = self.model(x)
        res = self.conv_block(pred)
        res_img = self.conv_block(x[:, 0:1, :, :])
        mask_vec = self.linear_block(res)
        img_vec = self.linear_block(res_img)
        return pred, mask_vec + img_vec  # self.weight_mask * mask_vec + (1 - self.weight_mask) * img_vec

    def forward(self, x_mlo, x_cc):
        pred_mlo, vec_mlo = self.forward_once(x_mlo)
        pred_cc, vec_cc = self.forward_once(x_cc)
        return pred_mlo, pred_cc, vec_mlo, vec_cc


def binarize_mask(mask, threshold=0.5):
    mask_sigm = torch.sigmoid(mask)
    mask_binar = torch.where(mask_sigm > threshold, torch.ones_like(mask_sigm), torch.zeros_like(mask_sigm))
    return mask_binar


