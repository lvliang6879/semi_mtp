#!/bin/bash

# ==============================
# 训练脚本：启动 SAR 图像半监督/微调实验
# 支持多卡分布式训练（DDP）
# ==============================

# --- 基础配置 ---
now=$(date +"%Y%m%d_%H%M%S")          # 当前时间戳，用于日志命名

# 数据集选择（三选一）
dataset='AIRPolarSARSeg'              # ✅ 当前使用：极化 SAR 分割数据集
# dataset='vaihingen'                 # 光学遥感数据集（Vaihingen）
# dataset='loveda'                    # 光学遥感数据集（LoveDA）

# 训练方法（主入口脚本）
method='main_finetune_sar'            # ✅ 微调 SAR 模型
# method='main_finetune_my_peft'      # 自定义 PEFT 微调, 加载预训练权重时需要修改 modality 超参，确定进行opt或sar的微调
# method='main_finetune_opt'          # ✅ 微调 OPT 模型

# 实验名称（用于区分不同预训练/策略）
exp='best_dinov3_vit_b_SemiM3P_48k'   # 基于 DINOv3 ViT-B + SemiM3P 伪标签的 48k 样本实验

# 数据划分（通常为 'all'，LoveDA 可用 'train'）
split='all'
# split='train'                       # 仅 LoveDA 需要指定 train/val/test

# --- 路径配置 ---
config="configs/${dataset}.yaml"                      # 数据集配置文件
# labeled_id_path=splits/$dataset/$split/labeled.txt  # 光学标注样本列表
labeled_id_path="splits/$dataset/$split/no_hh.txt"    # 已标注样本列表（SAR 特定：排除 HH 极化）
unlabeled_id_path="splits/$dataset/$split/unlabeled.txt"  # 未标注样本列表
save_path="exp/$dataset/$method/$exp/$split/"         # 实验结果保存路径

# 创建保存目录
mkdir -p "$save_path"

# --- GPU 设置 ---
export CUDA_VISIBLE_DEVICES=1,7       # 指定可见 GPU（例如使用第 1 和第 7 卡）

# --- 启动分布式训练 ---
# 参数说明：
#   $1 → nproc_per_node：每台机器使用的 GPU 数量（如 2）
#   $2 → master_port：DDP 通信端口（如 29500）
#   $3 → backbone：骨干网络类型（如 'dinov3_vitb'）
#   $4 → init_backbone：预训练权重路径（.pth 文件）
#   $5 → load：加载模式（如 'network' 表示加载完整模型）



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


# 启动指令： 
# bash scripts/main_finetune.sh 2 4567 dinov3_vit_b none network 全量微调
# bash scripts/main_finetune.sh 2 4567 dinov3_vit_b_adapter none network 外接adapter微调