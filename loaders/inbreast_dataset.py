from torch.utils.data import Dataset
import os
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import pandas as pd
import torch


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
