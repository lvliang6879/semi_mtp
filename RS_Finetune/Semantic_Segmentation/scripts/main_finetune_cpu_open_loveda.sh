#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

dataset='OpenEarthMap'
method='main_finetune_cpu_open_loveda'
exp='vit_b_upernet_s5'
# exp='vit_b_upernet_mae'
# split=all_512_256
split=trainval

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split/weak_aug_epoch_300/

mkdir -p $save_path
export CUDA_VISIBLE_DEVICES=0,1
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    --use_env $method.py \
    --backbone $3 \
    --init_backbone $4 \
    --load $5 \
    --config=$config --labeled-id-path $labeled_id_path --unlabeled-id-path $unlabeled_id_path \
    --save-path $save_path --port $2 2>&1 | tee $save_path/$now.log
