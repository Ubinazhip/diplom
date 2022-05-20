from tqdm import tqdm
import torch
import argparse
import dataset
import model
from utils import losses
import monai
from utils import losses
import statistics
from model import binarize_mask


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4, help="batch_size")
    parser.add_argument("--fold", type=int, default=0, help="fold number: 0, 1, 2, 3, 4")
    parser.add_argument('--transform_val', type=str, default=None,
                        help='path to the transform for validation yaml file')
    parser.add_argument('--model', type=str, default='unet', help='model name: unet')
    parser.add_argument('--loss_weight', type=float, default=0.5, help='weighted average of losses')
    parser.add_argument('--load_model', type=str, default=None, required=True, help='path to the pretrained model')
    args = parser.parse_args()
    return args


def inference(model, loader, criterion, metric, mode='train', vit=False):
    model.eval()
    tqdm_loader = tqdm(loader)
    running_loss = 0
    running_dice = 0
    for idx_batch, batch in enumerate(tqdm_loader):
        img_mlo, img_cc, label_mlo, label_cc = batch
        img_mlo, img_cc, label_mlo, label_cc = img_mlo.cuda(), img_cc.cuda(), label_mlo.cuda(), label_cc.cuda()
        with torch.no_grad():
            if vit:
                pred_mlo = model(img_mlo)
                loss = criterion(pred_mlo, label_mlo)
                dice = metric(torch.sigmoid(pred_mlo), label_mlo, per_image=True)
            else:
                pred_mlo, pred_cc = model(x_mlo=img_mlo, x_cc=img_cc)
                loss = 0.5 * criterion(pred_mlo, label_mlo) + 0.5 * criterion(pred_cc, label_cc)
                mlo_dice = metric(torch.sigmoid(pred_mlo), label_mlo, per_image=True)
                cc_dice = metric(torch.sigmoid(pred_cc), label_cc, per_image=True)
                dice = 0.5 * mlo_dice + 0.5 * cc_dice

            running_loss += loss.item() * img_mlo.size(0)
            running_dice += dice.item() * img_mlo.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_dice = running_dice / len(loader.dataset)
    print(f'{mode}: epoch loss = {epoch_loss:.4f}, dice = {epoch_dice:.4f}')
    return epoch_dice, epoch_loss


if __name__ == '__main__':
    args = parser()
    model = model.get_model(model_name=args.model)
    model = model.cuda()

    checkpoint = torch.load(args.load_model)['model_state_dict']

    model.load_state_dict(checkpoint)

    dataloaders = dataset.data_loader(fold=args.fold, batch_size=args.batch_size,
                                      train_transform=args.transform_val,
                                      val_transform=args.transform_val)

    criterion = losses.ComboLoss({'dice': 1, 'bce': 0})
    valid_metric = losses.dice_metric

    mean_metric, mean_loss = inference(model, dataloaders['test'], criterion=criterion, metric=valid_metric)
