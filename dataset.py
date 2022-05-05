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


class CustomDataset(Dataset):
    def __init__(self, df, root='/home/ibm_prod/data/cbis_ddsm_dcm/Mass/', mode='Train', transform=None):
        self.df = df
        if mode == 'Train':
            black_list = ['P_01062', 'P_01051', 'P_01112', 'P_01123']
            self.df = self.df[~self.df['id'].isin(black_list)]
        self.root = os.path.join(root, mode, f'{mode}_FULL')
        self.root_mask = os.path.join(root, mode, f'{mode}_MASK')
        if transform is None:
            self.transform = A.Compose([
                A.Resize(1024, 1024),
                ToTensorV2()
            ])
        else:
            self.transform = transform
        self.mode = mode
        self.dict_class = {'BENIGN': 0, 'MALIGNANT': 1, 'BENIGN_WITHOUT_CALLBACK': 0}
        self.dict_view = {'CC': 0, 'MLO': 1}

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

        img_path_mlo = os.path.join(self.root, f'{row.mlo}')
        img_path_cc = os.path.join(self.root, f'{row.cc}')
        img1 = self.read_img(img_path_mlo)
        img2 = self.read_img(img_path_cc)
        id_patient = row['id']
        mlo = row['mlo']
        side = mlo.split('_')[3]
        mode = mlo.split('_')[0]

        img_mask_mlo = glob.glob(os.path.join(self.root_mask, f'{mode}_{id_patient}_{side}_MLO_MASK*'))
        img_mask_mlo = img_mask_mlo[0]
        img_mask_cc = glob.glob(os.path.join(self.root_mask, f'{mode}_{id_patient}_{side}_CC_MASK*'))

        img_mask_cc = img_mask_cc[0]
        mask1 = self.read_img(img_mask_mlo, mask=True)
        mask2 = self.read_img(img_mask_cc, mask=True)

        if self.transform:
            transformed = self.transform(image=img1, mask=mask1, image1=img2, mask1=mask2)
            img1, mask1 = transformed['image'], transformed['mask']
            img2, mask2 = transformed['image1'], transformed['mask1']
        img1 = (img1).to(torch.float32)
        img1 = img1/255
        img1 = (img1 - torch.min(img1)) / (torch.max(img1) - torch.min(img1))

        img2 = (img2).to(torch.float32)
        img2 = img2/255
        img2 = (img2 - torch.min(img2)) / (torch.max(img2) - torch.min(img2))

        mask1 = (mask1).to(torch.float32)
        mask1 = mask1 / 255

        mask2 = (mask2).to(torch.float32)
        mask2 = mask2/255

        return img1, img2, mask1[None,], mask2[None,]


def data_loader(fold=0, batch_size=8, train_transform=None, val_transform=None):
    df_train = pd.read_csv(f'./folds/train{fold}.csv')
    df_val = pd.read_csv(f'./folds/val{fold}.csv')
    df_test = pd.read_csv(f'./folds/data_test.csv')

    if train_transform is not None:
        train_transform = A.load(train_transform, data_format='yaml')
    if val_transform is not None:
        val_transform = A.load(val_transform, data_format='yaml')
    data_train = CustomDataset(df_train, transform=train_transform)
    data_val = CustomDataset(df_val, transform=val_transform)
    data_test = CustomDataset(df_test, transform=val_transform, mode='Test')

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(data_val, batch_size=batch_size, shuffle=False, num_workers=8)
    test_loader = DataLoader(data_test, batch_size=batch_size, shuffle=False, num_workers=8)

    return train_loader, val_loader, test_loader


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