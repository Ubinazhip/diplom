#!/bin/sh

for fold in {2..2}
do
     CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 2 --epochs 150 --dice 1 --bce 1 --fold $fold --model unetr --stop_patience 20 --vit --scheduler_patience 6 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml --save_res_csv ./best_models/res_unetr.csv
done

for fold in {2..2}
do
     CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 2 --epochs 150 --dice 1 --fold $fold --model unetr --stop_patience 20 --vit --scheduler_patience 6 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml --save_res_csv ./best_models/res_unetr.csv
done

for fold in {2..2}
do
     CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 2 --epochs 150 --dice 1 --fold $fold --model unetr --stop_patience 20 --vit --scheduler_patience 6 --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml --save_res_csv ./best_models/res_unetr.csv
done

for fold in {2..2}
do
     CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 2 --epochs 150 --dice 2 --bce 1 --focal 1 --fold $fold --model unetr --stop_patience 20 --vit --scheduler_patience 6 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml --save_res_csv ./best_models/res_unetr.csv
done


for fold in {2..2}
do
     CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 2 --epochs 150 --dice 3 --bce 1 --focal 4 --fold $fold --model unetr --stop_patience 20 --vit --scheduler_patience 6 --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml --save_res_csv ./best_models/res_unetr.csv
done
