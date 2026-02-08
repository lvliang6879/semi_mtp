#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='pretrain_det'
method='sdp'
# method='s4p_mae_adaptive'
backbone=$3  # 从脚本参数获取

config=configs/${dataset}.yaml
# labeled_id_path=splits/$dataset/labeled/labeled_300.txt
labeled_id_path=splits/$dataset/labeled/labeled_ms.txt
unlabeled_id_path=splits/$dataset/unlabeled/RS4P-1M-final-1M-final.txt
save_path=exp/$method/$backbone/labeled_mota_ms/   # backbone 自动加入路径

mkdir -p $save_path
export CUDA_VISIBLE_DEVICES=4,5,6,7

python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    --use_env $method.py \
    --backbone $backbone \
    --init_backbone $4 \
    --decoder 'upernet' \
    --config=$config \
    --labeled-id-path $labeled_id_path \
    --save-path $save_path \
    --port $2 2>&1 | tee $save_path/$now.log
