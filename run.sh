#!/bin/sh


for i in {0..4}
do
     CUDA_VISIBLE_DEVICES=0 python3 runner.py --batch_size 4 --epochs 150 --dice 1 --bce 0 --fold $i --model unet_resnet34 --stop_patience 20 --scheduler_patience 10 --save_res_csv ./best_models/baseline_run.csv --lr 3e-4 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml
done

for i in {0..4}
do
     CUDA_VISIBLE_DEVICES=0 python3 runner.py --batch_size 4 --epochs 150 --dice 1 --bce 0 --fold $i --model unet_resnet50 --stop_patience 20 --scheduler_patience 10 --save_res_csv ./best_models/baseline_run.csv --lr 3e-4 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml
done

for i in {0..4}
do
     CUDA_VISIBLE_DEVICES=0 python3 runner.py --batch_size 4 --epochs 150 --dice 1 --bce 1 --fold $i --model unet_resnet34 --stop_patience 20 --scheduler_patience 10 --save_res_csv ./best_models/baseline_run.csv --lr 3e-4 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml
done

for i in {0..4}
do
     CUDA_VISIBLE_DEVICES=0 python3 runner.py --batch_size 4 --epochs 150 --dice 1 --bce 1 --fold $i --model unet_resnet50 --stop_patience 20 --scheduler_patience 10 --save_res_csv ./best_models/baseline_run.csv --lr 3e-4 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml
done
