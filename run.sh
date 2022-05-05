#!/bin/sh



CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 4 --epochs 100 --fold 0 --model unet --stop_patience 15 --scheduler_patience 7 --lr 3e-4 --loss_weight 1 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml

