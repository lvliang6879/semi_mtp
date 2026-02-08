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
from model.semseg.upernet import UperNet
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, color_map, intersectionAndUnionGPU
from PIL import Image
import cv2
import random
import logging
import torch.nn.functional as F
from collections import OrderedDict
from model.semseg.deeplabv3plus import DeepLabV3Plus


MAPPING = OrderedDict({
    'Impervious_surface': (255, 255, 255),
    'Building ': (0, 0, 255),
    'Low_vegetationc': (0, 255, 255),
    'Tree': (0, 255, 0),
    'Car': (255, 255, 0),
    'Others': (255, 0, 0),
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



def multi_scale_inference(model, img, scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.2]):
    """
    对输入图像进行多尺度推理，并融合不同尺度的预测结果。

    Args:
        model (nn.Module): 语义分割模型（应为 eval 状态）
        img (Tensor): 输入图像，形状为 [B, C, H, W]
        scales (list of float): 使用的缩放比例列表
    
    Returns:
        pred (Tensor): 最终类别预测图，形状为 [B, H, W]
    """
    model.eval()
    original_size = img.shape[-2:]  # (H, W)
    preds = []

    with torch.no_grad():
        for scale in scales:
            scaled_img = F.interpolate(img, scale_factor=scale, mode='bilinear', align_corners=True)
            scaled_pred = model(scaled_img)  # [B, C, H_s, W_s]
            scaled_pred = F.interpolate(scaled_pred, size=original_size, mode='bilinear', align_corners=True)
            preds.append(scaled_pred)

        fused_pred = torch.stack(preds, dim=0).mean(dim=0)  # [B, C, H, W]
        pred = fused_pred.argmax(dim=1)  # [B, H, W]

    return pred

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

def evaluate(model, loader, mode, cfg, ddp=False):
    model.eval()
    # assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()

    with torch.no_grad():

        for img, mask, id in loader:
            img = img.cuda()
            # mask = mask.cuda()
            x = img

            if mode == 'slide_window':
                b, _, h, w = x.shape    # 获取输入图像的尺寸 (batch, channels, height, width)
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()  # 用于存储最终预测结果
                size = cfg['crop_size']
                # step = int(size * 2 / 3)
                step = 512
                b = 0
                a = 0
                while (a <= int(h / step)):
                    while (b <= int(w / step)):
                        sub_input = x[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)]
                        # print("sub_input.shape", sub_input.shape)
                        pre = model(sub_input) 
                        # pre = net_process(model, sub_input, cfg)
                        final[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)] += pre
                        b += 1
                    b = 0
                    a += 1
                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                
                elif mode == 'resize':
                # 使用缩放方式进行预测
                    original_shape = img.shape[-2:]  # 保存原始图像的尺寸 (h, w)
                    # resized_x = F.interpolate(img, size=cfg['crop_size'], mode='bilinear', align_corners=True)
                    resized_x = F.interpolate(img, size=1024, mode='bilinear', align_corners=True)
                    resized_o = model(resized_x, cfg['dataset'])   
                    # 将预测结果复原到原始尺寸
                    o = F.interpolate(resized_o, size=original_shape, mode='bilinear', align_corners=True)
                    pred = o.argmax(dim=1)
                
                else:
                    pred = model(img).argmax(dim=1)
                    # pred = net_process(model, img, cfg).argmax(dim=1)
            
            mask = np.array(mask, dtype=np.int32)
            intersection, union, target, predict = intersectionAndUnion(pred.cpu().numpy(), mask, cfg['nclass'], cfg['ignore_index'])
            # intersection, union, target, predict = intersectionAndUnion(pred.cpu().numpy(), mask, cfg['nclass'], 255)

            if ddp:
                reduced_intersection = torch.from_numpy(intersection).cuda()
                reduced_union = torch.from_numpy(union).cuda()
                reduced_target = torch.from_numpy(target).cuda()

                dist.all_reduce(reduced_intersection)
                dist.all_reduce(reduced_union)
                dist.all_reduce(reduced_target)

                intersection_meter.update(reduced_intersection.cpu().numpy())
                union_meter.update(reduced_union.cpu().numpy())
            else:
                intersection_meter.update(intersection)
                union_meter.update(union)
                target_meter.update(target)
                predict_meter.update(predict)

    
        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
        accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10) * 100.0
        precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10) * 100.0
        F1_class = 2*(precise_class*accuracy_class) / (precise_class+accuracy_class)

        # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
        mIoU = np.mean(iou_class)
        mAcc = np.mean(accuracy_class)
        mF1 = np.mean(F1_class)
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc, iou_class, F1_class


def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/S5/configs/isaid_ori.yaml')
    parser.add_argument('--ckpt-path', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/semi_sep/exp/isaid_ori/supervised/vit_b/best.pth')
    parser.add_argument('--backbone', type=str, default='vit_b', required=False)
    parser.add_argument('--init_backbone', type=str, default='none', required=False)
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    model = UperNet(args, cfg)
    # model = DeepLabV3Plus(cfg) if args.backbone == 'r101' else UperNet(args, cfg)
    ckpt = torch.load(args.ckpt_path)['model']
    ckpt = remove_module_prefix(ckpt) if cfg['dataset'] != 'pascal' else ckpt
    model.load_state_dict(ckpt)
    model.cuda()
    print('Total params: {:.1f}M\n'.format(count_params(model)))

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=8, drop_last=False)
    
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    eval_mode = 'slide_window'
    # eval_mode = 'original'
    mIoU, mAcc, mF1, allAcc, iou_class, F1_class = evaluate(model, valloader, eval_mode, cfg)

    for (cls_idx, F1) in enumerate(F1_class):
        print('***** Evaluation ***** >>>> Class [{:} {:}] '
                    'F1: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], F1))
    print('***** Evaluation {} ***** >>>> MeanF1: {:.2f}\n'.format(eval_mode, mF1))

    for (cls_idx, IoU) in enumerate(iou_class):
        print('***** Evaluation ***** >>>> Class [{:} {:}] '
                    'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], IoU))
    print('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
    
    
if __name__ == '__main__':
    main()
