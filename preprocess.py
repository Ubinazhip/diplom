from torch.utils.data import DataLoader
#from utils import CustomDataset, get_model, validate, data_loader
import matplotlib.pyplot as plt
from IPython.display import clear_output
import numpy as np
import pandas as pd
import os

import os
import pandas as pd
from IPython.display import clear_output
import numpy as np
from tqdm import tqdm
import torch.nn as nn

from timm.models.efficientnet import *
import torchvision.models as models

import torch
import torch.nn.functional as F
import albumentations as A

import time

from albumentations.pytorch.transforms import ToTensorV2
import glob
import cv2

from tqdm import tqdm


def divide2(files):
    file_name1 = files[0].split('/')[-1]
    side1 = file_name1.split('_')[3]
    view1 = file_name1.split('_')[4]

    file_name2 = files[1].split('/')[-1]
    side2 = file_name2.split('_')[3]
    view2 = file_name2.split('_')[4]
    patient_id = file_name1.split('_')[1] + '_' + file_name1.split('_')[2]
    if side1 != side2:
        return None, None, None

    if view1 == 'MLO':
        mlo = file_name1
        cc = file_name2
    else:
        mlo = file_name2
        cc = file_name1

    return mlo, cc, patient_id


def divide3(files):
    file_name1 = files[0].split('/')[-1]
    side1 = file_name1.split('_')[3]
    view1 = file_name1.split('_')[4]
    patient_id = file_name1.split('_')[1] + '_' + file_name1.split('_')[2]

    mlo, cc = None, None
    for i in range(1, len(files)):
        file_name = files[i].split('/')[-1]
        side = file_name.split('_')[3]
        view = file_name.split('_')[4]

        if side == side1:
            if view == 'MLO':
                mlo = file_name
                cc = file_name1
            else:
                mlo = file_name1
                cc = file_name

    if mlo is None:
        if view == 'MLO':
            mlo = file_name
            cc = files[1].split('/')[-1]
        else:
            mlo = files[1].split('/')[-1]
            cc = file_name

    return mlo, cc, patient_id


def divide4(files):
    sides = []
    views = []
    file_names = []
    for file in files:
        file_name1 = file.split('/')[-1]
        file_names.append(file_name1)
        side1 = file_name1.split('_')[3]
        view1 = file_name1.split('_')[4]
        sides.append(side1)
        views.append(view1)

    patient_id = file_name1.split('_')[1] + '_' + file_name1.split('_')[2]
    visited = []
    mlos = []
    ccs = []
    for i in range(0, len(sides) - 1):
        for j in range(i + 1, len(sides)):
            if j not in visited:

                if sides[i] == sides[j]:
                    if views[i] == 'MLO':
                        mlo1 = file_names[i]
                        cc1 = file_names[j]

                    else:
                        mlo1 = file_names[j]
                        cc1 = file_names[i]
                    visited.append(j)
                    mlos.append(mlo1)
                    ccs.append(cc1)

    return mlos[0], ccs[0], mlos[1], ccs[1], patient_id


df = pd.read_csv('/home/ibm_prod/data/cbis_ddsm_dcm/csv_files/mass_case_description_test_set.csv')
file_ids = list(df.patient_id.unique())
path = '/home/ibm_prod/data/cbis_ddsm_dcm/Mass/Test/Test_FULL'


my_dict = {0: 0, 1: 0, 2: 0, 3: 0, 4:0}
a = 1
ids = []
mlos = []
ccs = []
for idx, i in enumerate(file_ids):
    files = glob.glob(os.path.join(path, f'Mass-Test_{i}*.png'))

    if len(files) == 2:
        mlo, cc, patient_id = divide2(files)
        if mlo is None:
            continue
        mlos.append(mlo)
        ccs.append(cc)
        ids.append(patient_id)
    if len(files) == 3:
        mlo, cc, patient_id = divide3(files)
        mlos.append(mlo)
        ccs.append(cc)
        ids.append(patient_id)
    if len(files) == 4 and a == 1:
        mlo1, cc1, mlo2, cc2, patient_id = divide4(files)
        mlos.append(mlo1)
        ccs.append(cc1)
        ids.append(patient_id)
        mlos.append(mlo2)
        ccs.append(cc2)
        ids.append(patient_id)

df_view = pd.DataFrame(list(zip(ids, mlos, ccs)),
               columns =['id', 'mlo', 'cc'])

for i in range(0, df_view.shape[0]):
    file_name_mlo = df_view.iloc[i]['mlo']
    file_name_cc = df_view.iloc[i]['cc']
    split_mlo = file_name_mlo.split('_')
    split_cc = file_name_cc.split('_')

    if split_mlo[2] != split_cc[2]:
        print(f'Upps, {i}, ids are not equal')
    if split_mlo[3] != split_cc[3]:
        print(f'Upps, {i}, sides are not equal')

    if split_mlo[4] == split_cc[4]:
        print(f'Upps, {i}, views are equal')


df = pd.read_csv('.folds/data.csv')
df_test = pd.read_csv('.folds/data_test.csv')
from sklearn.model_selection import GroupKFold
group_kfold = GroupKFold(n_splits=5)
idx = 0
for train_index, val_index in group_kfold.split(df, groups=df['id']):
    df.iloc[train_index].to_csv(f'./folds/train{idx}.csv', index=False)
    df.iloc[val_index].to_csv(f'./folds/val{idx}.csv', index=False)
    idx += 1