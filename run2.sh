#!/bin/sh


CUDA_VISIBLE_DEVICES=0,3 python3 runner.py --batch_size 3 --epochs 100 --fold 0 --model unetr --stop_patience 15 --scheduler_patience 7 --lr 3e-4 --loss_weight 0.9 --transform ./transforms/train_transform.yml --transform_val ./transforms/val_transform.yml



'''
#CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 4 --epochs 100 --fold 0 --model unetr --stop_patience 15 --scheduler_patience 7 --lr 3e-4 --loss_weight 0.9 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml

#CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 8 --epochs 100 --fold 0 --model resnet50 --stop_patience 15 --scheduler_patience 7 --lr 3e-4 --loss_weight 0.7 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml
#CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 8 --epochs 100 --fold 1 --model resnet50 --stop_patience 15 --scheduler_patience 7 --lr 3e-4 --loss_weight 0.7 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml
#CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 8 --epochs 100 --fold 2 --model resnet50 --stop_patience 15 --scheduler_patience 7 --lr 3e-4 --loss_weight 0.7 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml
#CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 8 --epochs 100 --fold 3 --model resnet50 --stop_patience 15 --scheduler_patience 7 --lr 3e-4 --loss_weight 0.7 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml
#CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 8 --epochs 15 --fold 1 --model albunet --stop_patience 7 --scheduler_patience 4
#CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 8 --epochs 15 --fold 2 --model albunet --stop_patience 7 --scheduler_patience 4
#CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 8 --epochs 15 --fold 3 --model albunet --stop_patience 7 --scheduler_patience 4
#CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 8 --epochs 15 --fold 4 --model albunet --stop_patience 7 --scheduler_patience 4
'''