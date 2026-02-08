#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# dataset='OpenEarthMap'
# dataset='potsdam'
dataset='rsseg'
# dataset='potsdam'
# dataset='UDD'
method='main_finetune_cpu_rsseg_single'
exp='vit_b_moe_upernet_s5_with_expert_open_68.57'
# exp='vit_h_moe_upernet_s5_with_expert_potsdam_miou_79.09_loveda_epoch3'
# exp='vit_l_moe_upernet_s5_with_expert_potsdam_miou_78.76_loveda_55.67'
# exp='vit_b_upernet_mae'
# split=all_512_256
split=all

config=configs/${dataset}.yaml
labeled_id_path=splits/OpenEarthMap/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split/

mkdir -p $save_path
export CUDA_VISIBLE_DEVICES=6,7
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
