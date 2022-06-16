from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import cv2
import glob
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pandas as pd
import torch


class CbisDataset(Dataset):
    """
    Data loader for cbis-ddsm dataset
    vit - transformers, if vit, then we concatenate MLO and CC along y axis
    so that we send patches of the MLO and CC to the transformer
    """
    def __init__(self, fold=0, root='/home/ibm_prod/projects/datasets/cbis_ddsm/Mass_png', mode='Train',
                 transform=None, vit=False):

        if mode == 'Train' or mode == 'val':
            self.df = pd.read_csv(f'./folds/cbis/{mode.lower()}{fold}.csv')
            mode = 'Train'
        elif mode == 'Test':
            self.df = pd.read_csv('./folds/cbis/test_data_cbis.csv')

        self.root = os.path.join(root, mode, f'{mode}_FULL')
        self.root_mask = os.path.join(root, mode, f'{mode}_MASK')
        if transform is None:
            self.transform = A.Compose([
                A.Resize(512, 512),
                ToTensorV2()
            ], additional_targets={'image1': 'image', 'mask1': 'mask'})
        else:
            self.transform = transform
        self.mode = mode
        self.vit = vit

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_mlo = cv2.imread(os.path.join(self.root, row['MLO']))
        img_cc = cv2.imread(os.path.join(self.root, row['CC']))

        mask_mlo_path = glob.glob(os.path.join(self.root_mask, row['MLO'].replace('FULL.png', 'MASK*')))[0]
        mask_cc_path = glob.glob(os.path.join(self.root_mask, row['CC'].replace('FULL.png', 'MASK*')))[0]

        mask_mlo = cv2.imread(mask_mlo_path, 0)
        mask_cc = cv2.imread(mask_cc_path, 0)

        if self.transform:
            transformed = self.transform(image=img_mlo, mask=mask_mlo, image1=img_cc, mask1=mask_cc)
            img_mlo, mask_mlo = transformed['image'], transformed['mask']
            img_cc, mask_cc = transformed['image1'], transformed['mask1']
        img_mlo = (img_mlo).to(torch.float32)
        img_mlo = img_mlo / 255
        img_mlo = (img_mlo - torch.min(img_mlo)) / (torch.max(img_mlo) - torch.min(img_mlo))

        img_cc = (img_cc).to(torch.float32)
        img_cc = img_cc / 255
        img_cc = (img_cc - torch.min(img_cc)) / (torch.max(img_cc) - torch.min(img_cc))

        mask_mlo = (mask_mlo).to(torch.float32)
        mask_mlo = mask_mlo / 255

        mask_cc = (mask_cc).to(torch.float32)
        mask_cc = mask_cc / 255
        if self.vit:
            img = torch.cat([img_mlo, img_cc], dim=1)
            mask = torch.cat([mask_mlo, mask_cc], dim=0)
            return img, img, mask[None,], mask[None,]

        else:
            return img_mlo, img_cc, mask_mlo[None,], mask_cc[None,]


class InBreast(Dataset):
    """
    vit - transformers, if vit, then we concatenate MLO and CC along y axis
    so that we send patches of the MLO and CC to the transformer
    """
    def __init__(self, fold=0, root='/home/ibm_prod/projects/datasets/INBreast/', mode='train', transform=None, vit=False):

        self.df = pd.read_csv(f'/home/ibm_prod/projects/diplom/folds/inbreast/{mode}{fold}.csv')
        self.root = os.path.join(root, 'AllPNGs')
        self.root_mask = os.path.join(root, 'extras', 'MassSegmentationMasks')
        if transform is None:
            self.transform = A.Compose([
                A.Resize(1024, 1024),
                ToTensorV2()
            ], additional_targets={'image1': 'image', 'mask1': 'mask'})
        else:
            self.transform = transform
        self.mode = mode
        self.vit = vit

    def __len__(self):
        return len(self.df)

    def read_img(self, path, mask=False):
        img = cv2.imread(path, 0)
        if not mask:
            img = np.stack([img, img, img], axis=0)
            img = np.moveaxis(img, 0, 2)
        return img

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path_mlo = os.path.join(self.root, f'{row.ML}')
        ml_file_id = row.ML.split('_')[0]
        img_path_cc = os.path.join(self.root, f'{row.CC}')
        cc_file_id = row.CC.split('_')[0]
        img1 = self.read_img(img_path_mlo)
        img2 = self.read_img(img_path_cc)

        mask_mlo = os.path.join(self.root_mask, f'{ml_file_id}_mask.png')

        mask_cc = os.path.join(self.root_mask, f'{cc_file_id}_mask.png')

        mask1 = self.read_img(mask_mlo, mask=True)
        mask2 = self.read_img(mask_cc, mask=True)

        if self.transform:
            transformed = self.transform(image=img1, mask=mask1, image1=img2, mask1=mask2)
            img1, mask1 = transformed['image'], transformed['mask']
            img2, mask2 = transformed['image1'], transformed['mask1']
        img1 = (img1).to(torch.float32)
        img1 = img1 / 255

        img2 = (img2).to(torch.float32)
        img2 = img2 / 255

        mask1 = (mask1).to(torch.float32)
        mask1 = mask1 / 255

        mask2 = (mask2).to(torch.float32)
        mask2 = mask2 / 255

        if self.vit:
            img = torch.cat([img1, img2], dim=1)
            mask = torch.cat([mask1, mask2], dim=0)
            return img, img, mask[None,], mask[None,]

        else:
            return img1, img2, mask1[None,], mask2[None,]


def data_loader(fold=0, batch_size=8, train_transform=None, val_transform=None, dataset='cbis', vit=False):

    if train_transform is not None:
        train_transform = A.load(train_transform, data_format='yaml')
    if val_transform is not None:
        val_transform = A.load(val_transform, data_format='yaml')
    if dataset == 'cbis':
        data_train = CbisDataset(fold=fold, transform=train_transform, mode='Train', vit=vit)
        data_val = CbisDataset(fold=fold, transform=val_transform, mode='val', vit=vit)
        data_test = CbisDataset(fold=fold, transform=val_transform, mode='Test', vit=vit)
    elif dataset == 'inbreast':
        data_train = InBreast(fold=fold, mode='train', transform=train_transform, vit=vit)
        data_val = InBreast(fold=fold, mode='val', transform=val_transform, vit=vit)
        data_test = InBreast(fold=fold, mode='val', transform=val_transform, vit=vit)
    else:
        raise NotImplementedError('No such dataset')

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=8)

    dataloaders = {'train': train_loader, 'val': val_loader, 'test': test_loader}
    return dataloaders


def get_transforms():
    transform = A.Compose([
        A.Resize(height=1050, width=1050, always_apply=True),
        A.RandomCrop(height=1024, width=1024, always_apply=True),
        A.VerticalFlip(p=0.3),
        A.OneOf([
            A.RandomGamma(),
            A.RandomBrightnessContrast(),
        ], p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            A.GridDistortion(),
        ], p=0.2),
        A.ShiftScaleRotate(rotate_limit=30),
        ToTensorV2()

    ], additional_targets={'image1': 'image', 'mask1': 'mask'})
    transform_val = A.Compose([
        A.Resize(height=1050, width=1050, always_apply=True),
        A.CenterCrop(height=1024, width=1024, always_apply=True),
        ToTensorV2()

    ], additional_targets={'image1': 'image', 'mask1': 'mask'})
    
    return transform, transform_val

