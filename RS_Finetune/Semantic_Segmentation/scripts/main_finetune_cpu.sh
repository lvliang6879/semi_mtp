#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# dataset='OpenEarthMap'
# dataset='IRSAMap'
# dataset='vaihingen'
dataset='AIRPolarSARSeg'
# dataset='isaid_ori'
# dataset='loveda'
# dataset='potsdam'
# dataset='OpenEarthMap'
# method='main_finetune_cpu_DINOv3'
method='main_finetune_cpu_sar'
# method='main_finetune_peft'
# exp='vit_l_selectivemae'
# exp='vit_b_upernet_S4P_my_mae_adaptive_mask_100k'
# exp='vit_b_upernet_S4P_labeled_300_80k'
# exp='vit_b_upernet_SEP_labeled_all'
# exp='vit_b_upernet_det_only_epoch12'
exp='dinov3_vit_b_vv_imagenet_normal'
# exp='best_dinov3_vit_b_multimodaltask_SARLO_36k'
# exp='vit_h_upernet_s4p'
# split=all_512_256
split=all
# split=train

config=configs/${dataset}.yaml
labeled_id_path=splits/$dataset/$split/labeled.txt
unlabeled_id_path=splits/$dataset/$split/unlabeled.txt
save_path=exp/$dataset/$method/$exp/$split/

mkdir -p $save_path
export CUDA_VISIBLE_DEVICES=0,3
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
