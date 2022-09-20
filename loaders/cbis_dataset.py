from torch.utils.data import Dataset
import os
import cv2
import glob
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
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
            self.df = pd.read_csv('../folds/cbis/test_data_cbis.csv')

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
