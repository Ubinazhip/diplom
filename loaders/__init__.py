from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from inbreast_dataset import InBreast
from cbis_dataset import CbisDataset


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

