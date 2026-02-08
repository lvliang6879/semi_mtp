import yaml
import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataset.test import SemiDataset
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, color_map, intersectionAndUnionGPU
from PIL import Image
import cv2
import random
import logging
import torch.nn.functional as F
from collections import OrderedDict
from model.semseg.upernet import UperNet
# from model.semseg.upernet_dinov3 import UperNet


# MAPPING = OrderedDict({
#     'Impervious_surface': (255, 255, 255),
#     'Building ': (0, 0, 255),
#     'Low_vegetationc': (0, 255, 255),
#     'Tree': (0, 255, 0),
#     'Car': (255, 255, 0)
# })


# MAPPING = OrderedDict({
#     'background': (255, 255, 255),
#     'building': (255, 0, 0),
#     'road': (255, 255, 0),
#     'water': (0, 0, 255),
#     'barren': (159, 129, 183),
#     'forest': (0, 255, 0),
#     'agriculture': (255, 195, 128),
# })

# MAPPING = OrderedDict({
#     'background': (0, 0, 0),
#     'ship': (0, 0, 63),
#     'storage_tank': (0, 191, 127),
#     'baseball_diamond': (0, 63, 0),
#     'tennis_court': (0, 63, 127),
#     'basketball_court': (0, 63, 191),
#     'ground_Track_Field': (0, 63, 255),
#     'bridge': (0, 127, 63),
#     'large_Vehicle': (0, 127, 127),
#     # 'large_Vehicle': (0, 0, 127),
#     'small_Vehicle': (0, 0, 127),
#     'helicopter': (0, 0, 191),
#     'swimming_pool': (0, 0, 255),
#     'roundabout': (0, 63, 63),
#     'soccer_ball_field': (0, 127, 191),
#     'plane': (0, 127, 255),
#     'harbor': (0, 100, 155),
# })


# MAPPING = OrderedDict({
#     'background': (0, 0, 0),
#     'ship': (0, 0, 63),
#     'storage_tank': (0, 191, 127),
#     'baseball_diamond': (0, 0, 0),
#     'tennis_court': (0, 0, 0),
#     'basketball_court': (0, 0, 0),
#     'ground_Track_Field': (0, 0, 0),
#     'bridge': (0, 0, 0),
#     # 'large_Vehicle': (0, 127, 127),
#     'large_Vehicle': (255, 255, 0),
#     'small_Vehicle': (255, 255, 0),
#     'helicopter': (0, 0, 191),
#     'swimming_pool': (0, 0, 0),
#     'roundabout': (0, 0, 0),
#     'soccer_ball_field': (0, 0, 0),
#     'plane': (0, 127, 255),
#     'harbor': (0, 100, 155),
# })


# MAPPING = OrderedDict({
#     "unknown": [0, 0, 0],
#     "Bareland": [128, 0, 0],
#     "Grass": [0, 255, 36],
#     "Pavement": [48, 148, 148],
#     "Road": [255, 255, 255],
#     "Tree": [34, 97, 38],
#     "Water": [0, 69, 255],
#     "Cropland": [75, 181, 73],
#     "buildings": [222, 31, 7],
# })

MAPPING = OrderedDict({
    'Background': (108, 98, 101),
    'Cropland': (255, 253, 145),
    'Tree': (32, 216, 109),
    'Grass': (1, 252, 119),
    'Water': (20, 197, 232),
    'Buildings': (210, 75, 97),
    'Road': (255, 200, 1)
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

def evaluate(model, loader, mode, cfg, ddp=False):
    model.eval()
    # assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()
    pred_output_dir = '/data1/users/zhengzhiyu/mtp_workplace/dataset/MOTA/split_ss_dota1_0/val/labels_IRSAMap/'
    rgb_output_dir = '/data1/users/zhengzhiyu/mtp_workplace/dataset/MOTA/split_ss_dota1_0/val/labels_IRSAMap_rgb/'
    os.makedirs(pred_output_dir, exist_ok=True)
    os.makedirs(rgb_output_dir, exist_ok=True)

    with torch.no_grad():
        for img_np, img, mask, id in loader:
            
            img = img.cuda()
            mask = mask.cuda()

            b, _, h, w = img.shape
                            
            if mode == 'sliding_window':
                final = torch.zeros(b, cfg['nclass'], h, w).cuda()  # 用于存储最终预测结果
                size = cfg['crop_size']
                # step = int(size * 2 / 3)
                step = int(size * 2 / 3)
                print('img.shape', img.shape)
                x = 0
                y = 0
                while (y <= int(h / step)):
                    while (x <= int(w / step)):
                        sub_input = img[:,:, min(y * step, h - size): min(y * step + size, h), min(x * step, w - size):min(x * step + size, w)]
                        mask = model(sub_input) 
                        final[:,:, min(y * step, h - size): min(y * step + size, h), min(x * step, w - size):min(x * step + size, w)] += mask
                        x += 1
                    x = 0
                    y += 1
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
                    resized_x = F.interpolate(img, size=cfg['crop_size'], mode='bilinear', align_corners=True)
                    resized_o = model(resized_x)   
                    # 将预测结果复原到原始尺寸
                    o = F.interpolate(resized_o, size=original_shape, mode='bilinear', align_corners=True)
                    pred = o.argmax(dim=1)
                
                else:
                    pred = model(img).argmax(dim=1)

                # loveda
                # rgb_mask = class_to_rgb_map(pred.squeeze().cpu())
            unique_classes = torch.unique(pred)

            # 打印图像 id 和包含的类别
            
            print("包含的类别索引:", unique_classes.tolist())
            image_name = os.path.splitext(os.path.basename(id[0]))[0]  
            print("图像 ID:", image_name)
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
            
            pred_mask = Image.fromarray(pred_np)
            pred_mask.save(os.path.join(pred_output_dir, f'{image_name}.png'))
            
            pred_rgb = class_to_rgb_map(pred_np)
            pred_rgb_pil = Image.fromarray(pred_rgb.astype(np.uint8))
            pred_rgb_pil.save(os.path.join(rgb_output_dir, f'{image_name}.png'))
                # vaihingen
                # rgb_pre = class_to_rgb_map(pred.squeeze().cpu())
                # print("id[0]", id[0][-8:])
                # rgb_pre_pil = Image.fromarray(rgb_pre.astype(np.uint8))
                # rgb_pre_output_path = os.path.join(model_output_dir, f'{id[0][-8:]}')
                # rgb_pre_pil.save(rgb_pre_output_path)

                # rgb_mask = class_to_rgb_map(mask.squeeze().cpu())
                # rgb_mask_pil = Image.fromarray(rgb_mask.astype(np.uint8))
                # rgb_mask_output_path = os.path.join(mask_output_dir, f'{id[0][-23:-4]}.png')
                # rgb_mask_pil.save(rgb_mask_output_path)
            # print("pred", pred.view(-1))
            # print("mask", mask.view(-1))
            # intersection, union, target = \
    #         #     intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], cfg['ignore_index'])
    #             intersection, union, target, predict = intersectionAndUnionGPU(pred, mask, cfg['nclass'], cfg['ignore_index'])
    #             intersection, union, target, predict = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy(), predict.cpu().numpy()
        
    #         if ddp:
    #             reduced_intersection = torch.from_numpy(intersection).cuda()
    #             reduced_union = torch.from_numpy(union).cuda()
    #             reduced_target = torch.from_numpy(target).cuda()

    #             dist.all_reduce(reduced_intersection)
    #             dist.all_reduce(reduced_union)
    #             dist.all_reduce(reduced_target)

    #             intersection_meter.update(reduced_intersection.cpu().numpy())
    #             union_meter.update(reduced_union.cpu().numpy())
    #         else:
    #             intersection_meter.update(intersection)
    #             union_meter.update(union)
    #             target_meter.update(target)
    #             predict_meter.update(predict)

    
    #     iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    #     accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10) * 100.0
    #     precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10) * 100.0
    #     F1_class = 2*(precise_class*accuracy_class) / (precise_class+accuracy_class)

    #     # iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    #     mIoU = np.mean(iou_class)
    #     mAcc = np.mean(accuracy_class)
    #     mF1 = np.mean(F1_class)
    #     allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    # return mIoU, mAcc, mF1, allAcc, iou_class, F1_class


def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/S5/configs/IRSAMap.yaml')
    # parser.add_argument('--ckpt-path', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/semi_sep/exp/OpenEarthMap/main_finetune_cpu_VDD/vit_h_upernet_mae_semi_sep_50000/all_wo_xbd/size_512_epoch_150_lr5e-5/vit_h_upernet_70.44.pth')
    # parser.add_argument('--ckpt-path', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/S5/exp/OpenEarthMap/main_finetune_cpu_DINOv3/vit_l_upernet_DINOV3/all/best.pth')
    parser.add_argument('--ckpt-path', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/S5/exp/IRSAMap/main_finetune_cpu/vit_h_upernet_s4p/all/best.pth')
    parser.add_argument('--backbone', type=str, default='vit_h', required=False)
    # parser.add_argument('--backbone', type=str, default='swin_l', required=False)
    parser.add_argument('--init_backbone', type=str, default='none', required=False)
    parser.add_argument('--image_size', type=int, default=512)
    parser.add_argument('--experts', type=int, default=4)
    # parser.add_argument('--tasks', nargs='+', default=['vaihingen', 'potsdam', 'OpenEarthMap', 'loveda', 'UDD', 'uavid'], type=str, help='List of dataset names')
    parser.add_argument('--tasks', nargs='+', default=['vaihingen', 'potsdam', 'OpenEarthMap', 'loveda']  , type=str, help='List of dataset names')
    parser.add_argument('--part', type=int, default=256)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    model = UperNet(args, cfg)

    # repo_dir = '/data1/users/zhengzhiyu/ssl_workplace/dinov3-main'
    # backbone = torch.hub.load(repo_dir, 'dinov3_vitl16', source='local', weights='/data1/users/zhengzhiyu/ssl_workplace/UniMatch-V2-main/pretrained/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth')
    # model = UperNet(args, cfg, backbone)

    ckpt = torch.load(args.ckpt_path)['model']
    ckpt = remove_module_prefix(ckpt)
    model.load_state_dict(ckpt)
    model.cuda()
    print('Total params: {:.1f}M\n'.format(count_params(model)))

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=8, drop_last=False)
    
    # eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    eval_mode = 'sliding_window'
    # eval_mode = 'original'
    # eval_mode = 'resize'
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
