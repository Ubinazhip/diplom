#!/bin/sh

for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 2 --bce 1 --fold $fold --loss_type dct_encoder --loss_weight 0.9 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/dct_encoder_run.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done


for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 2 --bce 1 --fold $fold --loss_type dct_encoder --loss_weight 0.8 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/dct_encoder_run.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done


for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 2 --bce 1 --fold $fold --loss_type dct_encoder --loss_weight 0.7 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/dct_encoder_run.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done





for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 1 --fold $fold --loss_type dct_encoder --loss_weight 0.9 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/dct_encoder_run.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done


for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 1 --fold $fold --loss_type dct_encoder --loss_weight 0.8 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/dct_encoder_run.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done


for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 1 --fold $fold --loss_type dct_encoder --loss_weight 0.7 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/dct_encoder_run.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done


for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 1 --fold $fold --loss_type dct_encoder --loss_weight 0.9 --model $model --stop_patience 20 --scheduler_patience 9 --save_res_csv ./best_models/dct_encoder_run.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done


for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 2 --bce 1 --fold $fold --loss_type dct_encoder --loss_weight 0.8 --model $model --stop_patience 20 --scheduler_patience 9 --save_res_csv ./best_models/dct_encoder_run.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done
