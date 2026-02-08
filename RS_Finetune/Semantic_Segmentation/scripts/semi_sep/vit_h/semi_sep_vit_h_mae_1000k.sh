#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='pretrain'
method='semi_sep'
exp='vit_h_upernet_mae'
backbone='vit_h'
init_backbone='mae'

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/labeled/labeled.txt
unlabeled_id_path=splits/$dataset/unlabeled/millionseg_1000k.txt
save_path=exp/$dataset/$method/$exp/millionseg_1000k/

mkdir -p $save_path
export CUDA_VISIBLE_DEVICES=4,5,6,7
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    --use_env $method.py \
    --backbone $backbone \
    --init_backbone $init_backbone \
    --decoder 'upernet' \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log