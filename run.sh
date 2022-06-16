#!/bin/sh

for fold in {0..4}
do
     CUDA_VISIBLE_DEVICES=0,1 python3 runner.py --batch_size 2 --dataset inbreast --epochs 1 --dice 1 --bce 1 --fold $fold --model unetr --stop_patience 20 --vit --scheduler_patience 6 --transform ./transforms/train_transform512.yml --transform_val ./transforms/val_transform512.yml --save_res_csv extra.csv
done

for model in unetpp_efficientnet-b3
do
  for fold in 2
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 1 --fold $fold --dataset inbreast --loss_type axis_sum --loss_weight 0.9 --model $model --stop_patience 25 --scheduler_patience 12 --save_res_csv ./extra.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done

