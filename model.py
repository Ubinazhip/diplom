import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from monai.networks.nets import UNet,UNETR
from models import ternausnets
from models.selim_zoo.unet import Resnet


def get_model(model_name='unet'):
    if model_name == 'unet':
        model = UNet(
            spatial_dims=2,
            in_channels=3,
            out_channels=1,
            channels=(4, 8, 16, 32, 64, 128, 64, 32),
            strides=(2, 2, 2, 2, 2, 2, 2),
        )
        model = Model(model)#MyModel(model, img_size=512)
    elif model_name == 'albunet':
        model = ternausnets.AlbuNet()
    elif model_name == 'resnet50':
        model = Resnet(seg_classes=1, backbone_arch='resnet50')
        model = MyModel(model, img_size=512)
    elif model_name == 'unetr':
        model = UNETR(in_channels=3, out_channels=1, img_size=512, feature_size=16, norm_name='batch', spatial_dims=2)
        model = MyModel(model, img_size=512)
    else:
        print('No such model in our zoo...')
        exit()
    print(f'The model {model_name} is loaded')
    return model


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

        vec_mlo = torch.sum(binarize_mask(pred_mlo), dim=(1, 2))
        vec_cc = torch.sum(binarize_mask(pred_cc), dim=(1, 2))

        #vec_mlo = (vec_mlo - vec_mlo.mean()) / vec_mlo.std()
        #vec_cc = (vec_cc - vec_cc.mean()) / vec_cc.std()

        return pred_mlo, pred_cc, vec_mlo, vec_cc