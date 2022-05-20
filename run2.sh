#!/bin/sh

for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 2 --bce 1 --fold $fold --loss_type axis_sum --loss_weight 0.99 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/axis_sum_run2.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done

for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 2 --bce 1 --fold $fold --loss_type axis_sum --loss_weight 0.999 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/axis_sum_run2.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done

for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 2 --bce 1 --fold $fold --loss_type axis_sum --loss_weight 0.9999 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/axis_sum_run2.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done


for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 1 --bce 1 --focal 1 --fold $fold --loss_type axis_sum --loss_weight 0.99 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/axis_sum_run2.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done

for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 1 --bce 1 --focal 1 --fold $fold --loss_type axis_sum --loss_weight 0.999 --model $model --stop_patience 12 --scheduler_patience 5 --save_res_csv ./best_models/axis_sum_run2.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done


for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 1 --bce 1 --focal 1 --fold $fold --loss_type axis_sum --loss_weight 0.999 --model $model --stop_patience 20 --scheduler_patience 9 --save_res_csv ./best_models/axis_sum_run2.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done

for model in fpn_densenet121 fpn_dpn68 fpn_efficientnet-b3 fpn_resnet34 fpn_resnet50
do
  for fold in {0..4}
  do
       CUDA_VISIBLE_DEVICES=3 python3 runner.py --batch_size 3 --epochs 150 --dice 1 --bce 1 --focal 1 --fold $fold --loss_type axis_sum --loss_weight 0.99 --model $model --stop_patience 20 --scheduler_patience 9 --save_res_csv ./best_models/axis_sum_run2.csv --transform ./transforms/aggressive.yml --transform_val ./transforms/aggressive_val.yml
  done
done
