import yaml
import argparse
import numpy as np
from tqdm import tqdm
import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from dataset.test import SemiDataset
# from model.semseg.upernet import UperNet
from model.semseg.s4_mae import UperNet
from util.classes import CLASSES
from util.utils import count_params, AverageMeter, intersectionAndUnion, color_map, intersectionAndUnionGPU
from PIL import Image
import cv2
import random
import logging
import torch.nn.functional as Fimport torch.nn.functional as F
from collections import OrderedDict
from s4_pretrain1 import validation


MAPPING = OrderedDict({
    'background': (0, 0, 0),
    'ship': (0, 0, 63),
    'storage_tank': (0, 191, 127),
    'baseball_diamond': (0, 63, 0),
    'tennis_court': (0, 63, 127),
    'basketball_court': (0, 63, 191),
    'ground_Track_Field': (0, 63, 255),
    'bridge': (0, 127, 63),
    'large_Vehicle': (0, 127, 127),
    'small_Vehicle': (0, 0, 127),
    'helicopter': (0, 0, 191),
    'swimming_pool': (0, 0, 255),
    'roundabout': (0, 63, 63),
    'soccer_ball_field': (0, 127, 191),
    'plane': (0, 127, 255),
    'harbor': (0, 100, 155),
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

def set_seeds(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



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


def visualize_masked_tokens(img_np, ids_keep, patch_size, save_dir, filenames):
    """
    可视化并保存 masked token 区域
    """
    os.makedirs(save_dir, exist_ok=True)
    B, H, W, _ = img_np.shape
    # print("img_np.shape", img_np.shape)
    L = (H // patch_size) * (W // patch_size)
    # print("L", L)

    all_idx = torch.arange(L, device=ids_keep.device).unsqueeze(0).repeat(B, 1)
    ids_masked = torch.ones_like(all_idx, dtype=torch.bool)
    ids_masked.scatter_(1, ids_keep, False)

    for b in range(B):
        img = np.array(img_np[b])
        filename = os.path.basename(filenames[b])
        mask = ids_masked[b].view(H // patch_size, W // patch_size).cpu().numpy()
        mask_up = np.kron(mask, np.ones((patch_size, patch_size)))

        vis_img = (img.copy() * 255 if img.max() <= 1 else img.copy()).astype(np.uint8)
        overlay = vis_img.copy()
        overlay[mask_up.astype(bool)] = [255, 0, 0]  # 红色覆盖 mask 区域
        alpha = 0.5
        vis_img = (vis_img * (1 - alpha) + overlay * alpha).astype(np.uint8)

        save_path = os.path.join(save_dir, filename)
        Image.fromarray(vis_img).save(save_path)
        print(f"✅ Saved masked token visualization: {save_path}")

def save_prediction_rgb(pred_rgb, save_dir, filenames):
    """
    保存预测结果 RGB 可视化
    """
    filename = os.path.basename(filenames[0])
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    Image.fromarray(pred_rgb.astype(np.uint8)).save(save_path)
    print(f"✅ Saved prediction visualization: {save_path}")


def evaluate(model, loader, mode, cfg, ddp=False):
    model.eval()
    # assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()

    with torch.no_grad():

        for img_np, img, mask, id in loader:
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
                    # pred = model(img).argmax(dim=1)
                    pred, _ = model(img, mask_ratio=0.75)
                    # pred, ids_keep = model(img, pred_w=pred_w, mask_ratio=0.5)
                    pred = pred.argmax(dim=1)
                    # pred, ids_keep = model(img, mask_ratio=0.25)
                    # pred = pred.argmax(dim=1)
                    # pred_np = pred.squeeze().cpu().numpy().astype(np.uint8)
                    # pred_rgb = class_to_rgb_map(pred_np)

                    # visualize_masked_tokens(
                    #     img_np=img_np, 
                    #     ids_keep=ids_keep, 
                    #     patch_size=16, 
                    #     save_dir="/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/visual/class_entropy/mask_ratio_0.5", 
                    #     filenames=id
                    # )
                    # pred = net_process(model, img, cfg).argmax(dim=1)


                    # save_prediction_rgb(
                    #     pred_rgb, 
                    #     save_dir="/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/visual/pred", 
                    #     filenames=id
                    # )

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
        if cfg['dataset'] == 'pretrain':
            mIoU = np.mean(iou_class[1:])
            mAcc = np.mean(accuracy_class[1:])
            mF1 = np.mean(F1_class[1:])
            allAcc = sum(intersection_meter.sum[1:]) / (sum(target_meter.sum[1:]) + 1e-10)
        else:
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            mF1 = np.mean(F1_class)
            allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    return mIoU, mAcc, mF1, allAcc, iou_class, F1_class


def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/configs/pretrain.yaml')
    parser.add_argument('--ckpt-path', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/exp/s4_pretrain1/vit_b/rs4p-1m-top1-0.85/vit_b_s4p_upernet_120k.pth')
    parser.add_argument('--backbone', type=str, default='vit_mae_b', required=False)
    parser.add_argument('--init_backbone', type=str, default='none', required=False)
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    model = UperNet(args, cfg)
    # model = DeepLabV3Plus(cfg) if args.backbone == 'r101' else UperNet(args, cfg)
    ckpt = torch.load(args.ckpt_path)['model']
    ckpt = remove_module_prefix(ckpt) if cfg['dataset'] != 'pascal' else ckpt
    # model.load_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    model.cuda()
    print('Total params: {:.1f}M\n'.format(count_params(model)))

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=8, drop_last=False)
    
    eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    # eval_mode = 'slide_window'
    eval_mode = 'original'
    mIoU, mAcc, mF1, allAcc, iou_class, F1_class = evaluate(model, valloader, eval_mode, cfg)
    # results = validation(model, valloader, cfg, eval_mode)

    # for (cls_idx, F1) in enumerate(F1_class):
    #     print('***** Evaluation ***** >>>> Class [{:} {:}] '
    #                 'F1: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], F1))
    # print('***** Evaluation {} ***** >>>> MeanF1: {:.2f}\n'.format(eval_mode, mF1))

    # for (cls_idx, IoU) in enumerate(results['iou_class']):
        # print('***** Evaluation ***** >>>> Class [{:} {:}] '
                    # 'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], IoU))
    # print('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, results['mIoU']))
    

    for (cls_idx, IoU) in enumerate(iou_class):
        print('***** Evaluation ***** >>>> Class [{:} {:}] '
                    'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], IoU))
    print('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
    
    
if __name__ == '__main__':
    set_seeds(2026)
    main()
