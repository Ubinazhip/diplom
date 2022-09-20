import torch
import segmentation_models_pytorch as smp
from monai.networks.nets import UNETR


def get_model(model_name='unet', encoder='resnet50', in_channels=3, classes=1):
    if model_name == 'unet':
        model = smp.Unet(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == 'unetpp':
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes,
        )
    elif model_name == 'fpn':
        model = smp.FPN(
            encoder_name=encoder,
            encoder_depth=5,
            encoder_weights='imagenet',
            decoder_pyramid_channels=256,
            decoder_segmentation_channels=128,
            in_channels=in_channels,
            classes=classes)
    elif model_name == 'unetr':
        model = UNETR(in_channels=3, out_channels=1,
                      img_size=(1024, 512), feature_size=16,
                      hidden_size=768,
                      mlp_dim=3072, num_heads=12, pos_embed='conv',
                      norm_name='batch', conv_block=True, res_block=True,
                      dropout_rate=0.0, spatial_dims=2)
        return model
    else:
        raise NotImplementedError('No such model in our zoo...')
    model = Model(model)
    print(f'The model {model_name}, with encoder {encoder} is loaded')
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


def binarize_mask(mask, threshold=0.5):
    mask_sigm = torch.sigmoid(mask)
    mask_binar = torch.where(mask_sigm > threshold, torch.ones_like(mask_sigm), torch.zeros_like(mask_sigm))
    return mask_binar


