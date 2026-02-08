#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

# 参数设置
dataset='rsseg'  # 可选数据集名用于指定 config.yaml
method='main_finetune_cpu_rsseg_vdd_loveda'  # 主训练脚本名（不含 .py）
# exp='vit_b_moe_wo_shared_upernet_s5'
# exp='vit_b_moe_upernet_s5'
exp='vit_l_moe_upernet_s5'
# exp='vit_b_upernet_s5'
split=all

# 配置路径
config=configs/${dataset}.yaml

# 六个数据集的 ID 路径
vaihingen_id_path=splits/vaihingen/$split/labeled.txt
potsdam_id_path=splits/potsdam/$split/labeled.txt
open_id_path=splits/OpenEarthMap/$split/labeled.txt
loveda_id_path=splits/loveda/$split/labeled.txt
udd_id_path=splits/UDD/$split/labeled.txt
vdd_id_path=splits/VDD/$split/labeled.txt

# 模型保存路径
save_path=exp/$dataset/$method/$exp/$split/part_256_75k/
mkdir -p $save_path

# GPU 设置（可根据需要更改）
export CUDA_VISIBLE_DEVICES=2,3,4,5

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
    --vaihingen-id-path $vaihingen_id_path \
    --potsdam-id-path $potsdam_id_path \
    --open-id-path $open_id_path \
    --loveda-id-path $loveda_id_path \
    --udd-id-path $udd_id_path \
    --vdd-id-path $vdd_id_path \
    --save-path $save_path \
    --port $2 2>&1 | tee $save_path/$now.log
