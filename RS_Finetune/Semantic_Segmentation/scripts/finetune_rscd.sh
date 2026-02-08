#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# 参数设置
dataset='rscd'  # 可选数据集名用于指定 config.yaml
method='main_finetune_CD_oscd_levir'  # 主训练脚本名（不含 .py）
# exp='vit_b_moe_wo_shared_upernet_s5'
# exp='vit_b_moe_upernet_s5'
exp='vit_b_moe_s5_unet_whu_oscd_parts_64'
# exp='vit_b_upernet_s5_rscd_whu'
split=all

# 配置路径
config=configs/${dataset}.yaml

# 六个数据集的 ID 路径
levir_id_path=splits/levir/$split/labeled.txt
whu_id_path=splits/whu/$split/labeled.txt
oscd_id_path=splits/oscd/$split/labeled.txt

# 模型保存路径
save_path=exp/$dataset/$method/$exp/$split/
mkdir -p $save_path

# GPU 设置（可根据需要更改）
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4,7

# 启动分布式训练
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    --master_addr=localhost \
    --master_port=$2 \
    --use_env $method.py \
    --backbone $3 \
    --init_backbone $4 \
    --load $5 \
    --config=$config \
    --levir-id-path $levir_id_path \
    --whu-id-path $whu_id_path \
    --oscd-id-path $oscd_id_path \
    --save-path $save_path \
    --port $2 2>&1 | tee $save_path/$now.log
