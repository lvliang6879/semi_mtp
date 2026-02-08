import yaml
import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from dataset.test import SemiDataset
from model.semseg.upernet_rsseg import UperNet
# from model.semseg.upernet import UperNet
# from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, color_map, intersectionAndUnionGPU
from PIL import Image
import cv2
import random
import logging
import torch.nn.functional as F
from collections import OrderedDict
# from model.semseg.models_samrs import SemsegFinetuneFramework
# from fvcore.nn import FlopCountAnalysis, parameter_count
from fvcore.nn import FlopCountAnalysis, parameter_count_table


MAPPING = OrderedDict({
    'background': (255, 255, 255),
    'building': (255, 0, 0),
    'road': (255, 255, 0),
    'water': (0, 0, 255),
    'barren': (159, 129, 183),
    'forest': (0, 255, 0),
    'agriculture': (255, 195, 128),
})

class_to_rgb = {idx: value for idx, value in enumerate(MAPPING.values())}

def class_to_rgb_map(image):
    # 转换类别索引为 RGB 颜色值
    h, w = image.shape
    # 创建新的 RGB 图像
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)

    # 将每个像素的类别索引映射到 RGB 颜色值
    for cls, rgb in class_to_rgb.items():
        mask = (image == cls)
        rgb_image[mask] = rgb

    return rgb_image



def remove_module_prefix(state_dict):
    """Remove 'module.' prefix from saved weights in multi-GPU training"""
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_key = k[len('module.'):]
        else:
            new_key = k
        new_state_dict[new_key] = v

    return new_state_dict




def evaluate_params_and_flops(model, image_size):
    # 统计参数
    params = parameter_count(model)
    total_params = sum(params.values())
    print("✅ Total Parameters: {:.2f} M".format(total_params / 1e6))

    # 统计 FLOPs
    dummy_input = torch.randn(1, 3, image_size, image_size).cuda()
    flops = FlopCountAnalysis(model, dummy_input)
    print("✅ Total FLOPs: {:.2f} G".format(flops.total() / 1e9))

def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/S5/configs/rsseg.yaml')
    parser.add_argument('--ckpt-path', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/S5/exp/rsseg/main_finetune_cpu_rsseg_four/vit_b_moe_upernet_s5/all/part_256_wo_loveda/epoch_075_miou_78.0817.pth')
    parser.add_argument('--backbone', type=str, default='vit_b_moe', required=False)
    parser.add_argument('--init_backbone', type=str, default='none', required=False)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--experts', type=int, default=4)
    # parser.add_argument('--tasks', nargs='+', default=['vaihingen', 'potsdam', 'OpenEarthMap', 'loveda', 'UDD', 'uavid'], type=str, help='List of dataset names')
    parser.add_argument('--tasks', nargs='+', default=['vaihingen', 'potsdam', 'OpenEarthMap', 'loveda']  , type=str, help='List of dataset names')
    parser.add_argument('--part', type=int, default=512)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    model = DeepLabV3Plus(cfg) if args.backbone == 'r101' else UperNet(args, cfg)
    # model = model.cuda()
    # ckpt = torch.load(args.ckpt_path)['model']
    # ckpt = remove_module_prefix(ckpt)
    # model.load_state_dict(ckpt)
    model.eval()
    input_tensor = torch.randn(1, 3, 512, 512)

    # 计算 FLOPs
    flops = FlopCountAnalysis(model, input_tensor)
    print("FLOPs: {:.2f} G".format(flops.total() / 1e9))

    # 计算参数量
    print(parameter_count_table(model))

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    if hasattr(model, 'backbone'):
        encoder_params = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    else:
        encoder_params = 0.0

    print(f"Total Parameters: {total_params:.2f} MB")
    print(f"Encoder Parameters (backbone): {encoder_params:.2f} MB")


if __name__ == '__main__':
    main()