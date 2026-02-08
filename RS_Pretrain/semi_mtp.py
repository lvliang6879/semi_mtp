import os
import time
import yaml
import pprint
import logging
import argparse
from pathlib import Path
from tqdm import tqdm

import random
import numpy as np
import cv2
import PIL.Image as Image

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

from dataset.semi_mtp import SemiDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, init_log, intersectionAndUnionGPU
from util.dist_helper import setup_distributed
from util.train_utils import DictAverageMeter, confidence_weighted_loss, cutmix_img_, cutmix_mask
# from model.ordet.orcnn import ORCNN
from model.ordet.multi_task_model import ORCNN

from mmengine.optim import build_optim_wrapper
from mmdet.structures import DetDataSample
from mmrotate.structures import RotatedBoxes
from mmengine.structures import InstanceData
from typing import List, Dict, Optional, Tuple, Union
from mmengine.utils import is_list_of
from mmengine.optim import build_optim_wrapper
from util.train_utils import build_param_scheduler, scheduler_after_train_iter
import math
# from mmcv_custom.layer_decay_optimizer_constructor_vit import *
# -----------------------
# Utils
# -----------------------
def set_seeds(seed=2025):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False


def save_checkpoint(state, backbone, save_dir: Path, backbone_name: str, suffix: str):
    """统一的 checkpoint 保存函数."""
    torch.save(state, save_dir / f'{backbone_name}_s4p_upernet.pth')
    torch.save(backbone, save_dir / f'{backbone_name}_s4p.pth')





def build_data_samples(imgs, targets):
    """
    imgs: Tensor [B, 3, H, W]
    targets: List[Dict], each may contain:
        - for labeled: 'boxes', 'labels', 'img_id', 'file_name', 'ori_shape', etc.
        - for unlabeled: only meta fields (no 'boxes'/'labels')
    Returns: List[DetDataSample]
    """
    B, C, H, W = imgs.shape
    device = imgs.device
    data_samples = []

    for i in range(B):
        ds = DetDataSample()
        target = targets[i]

        # === Metainfo (always present) ===
        img_id = target.get("img_id", i)
        file_name = target.get("file_name", f"{i}.png")
        img_path = target.get("img_path", "")
        ori_h, ori_w = target.get("ori_shape", (H, W))
        scale_factor = target.get("scale_factor", (1.0, 1.0))

        ds.set_metainfo({
            "img_shape": (H, W),
            "pad_shape": (H, W, C),
            "ori_shape": (ori_h, ori_w),
            "scale_factor": scale_factor,
            "batch_input_shape": (H, W),
            "img_id": img_id,
            "file_name": file_name,
            "img_path": img_path,
        })

        # === GT Instances: handle missing boxes/labels ===
        if "boxes" in target and "labels" in target:
            boxes = target["boxes"]
            labels = target["labels"]
            n = min(boxes.shape[0], labels.shape[0])
            boxes = boxes[:n]
            labels = labels[:n]
        else:
            # Unlabeled image: no annotations
            boxes = torch.empty((0, 5), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.long)
            n = 0

        # Move to device
        boxes = boxes.to(device)
        labels = labels.to(device)

        gt_inst = InstanceData()
        if n > 0:
            gt_inst.bboxes = RotatedBoxes(boxes)
            gt_inst.labels = labels
        else:
            gt_inst.bboxes = RotatedBoxes(torch.empty((0, 5), dtype=torch.float32, device=device))
            gt_inst.labels = torch.empty((0,), dtype=torch.long, device=device)

        # === Ignored Instances (always empty in semi-supervised) ===
        ign_inst = InstanceData()
        ign_inst.bboxes = RotatedBoxes(torch.empty((0, 5), dtype=torch.float32, device=device))
        ign_inst.labels = torch.empty((0,), dtype=torch.long, device=device)

        # Assign to data sample
        ds.gt_instances = gt_inst
        ds.r_gt_instances = gt_inst
        ds.ignored_instances = ign_inst
        ds.r_ignored_instances = ign_inst

        data_samples.append(ds)

    return data_samples


def det_collate_fn(batch):
    imgs, masks, targets = zip(*batch)         # tuple of length B
    imgs = torch.stack(imgs, dim=0)     # [B,3,H,W]
    masks = torch.stack(masks, dim=0)   # [B,1,H,W]
    return imgs, masks, list(targets)

def det_collate_fn_unsup(batch):
    """用于无监督数据：(img_w, img_s, target)"""
    img_w, img_s, ignore_masks, targets = zip(*batch)
    img_w = torch.stack(img_w, dim=0)
    img_s = torch.stack(img_s, dim=0)
    ignore_masks = torch.stack(ignore_masks, dim=0)   # [B,1,H,W]
    return img_w, img_s, ignore_masks, list(targets)

def sum_loss_dict(loss_dict):
    """
    Recursively sum all tensor losses in a (possibly nested) dict.
    """
    total_loss = 0.0
    for v in loss_dict.values():
        if isinstance(v, torch.Tensor):
            total_loss = total_loss + v
        elif isinstance(v, dict):
            total_loss = total_loss + sum_loss_dict(v)
        else:
            # 忽略非 tensor / 非 dict（极少见）
            pass
    return total_loss


def flatten_loss_dict(loss_dict, prefix=''):
    """
    Flatten nested loss dict for logging.
    """
    log_info = {}
    for k, v in loss_dict.items():
        name = f"{prefix}{k}" if prefix == '' else f"{prefix}.{k}"
        if isinstance(v, torch.Tensor):
            log_info[name] = v.detach().item()
        elif isinstance(v, dict):
            log_info.update(flatten_loss_dict(v, prefix=name))
    return log_info
# -----------------------
# Main Training
# -----------------------

def parse_losses(losses: List[Dict[str, torch.Tensor]]):
    """Parses the raw outputs (losses) of the network.

    Args:
        losses (dict): Raw output of the network, which usually contain
            losses and other necessary information.

    Returns:
        tuple[Tensor, dict]: There are two elements. The first is the
        loss tensor passed to optim_wrapper which may be a weighted sum
        of all losses, and the second is log_vars which will be sent to
        the logger.
    """
    # pprint(losses)
    loss_sum = 0
    # loss_datasets = []

    log_vars = []
    for loss_name, loss_value in losses.items():
        if isinstance(loss_value, torch.Tensor):
            log_vars.append([loss_name, loss_value.mean()])
        elif is_list_of(loss_value, torch.Tensor):
            log_vars.append(
                [loss_name, sum(_loss.mean() for _loss in loss_value)])
        else:
            raise TypeError(
                f'{loss_name} is not a tensor or list of tensors')
    # print(log_vars)
    loss = sum(value for key, value in log_vars if 'loss' in key)

    loss_sum += loss

    # loss_datasets.append(loss)

    return loss_sum

def extract_pseudo_labels_for_training(pseudo_data_samples, score_thr=0.7, max_num_per_img=100):
    new_data_samples = []
    for ds in pseudo_data_samples:
        # 优先使用标准字段 pred_instances
        if hasattr(ds, 'pred_instances'):
            pred = ds.pred_instances
        elif hasattr(ds, 'r_pred_instances'):
            pred = ds.r_pred_instances
        else:
            raise AttributeError("No prediction instances found in DetDataSample!")
        keep = pred.scores > score_thr

        # 处理 bboxes：确保是 Tensor 并转为 RotatedBoxes
        raw_bboxes = pred.bboxes
        if isinstance(raw_bboxes, RotatedBoxes):
            bboxes_tensor = raw_bboxes.tensor
        else:
            bboxes_tensor = raw_bboxes
        assert bboxes_tensor.size(-1) == 5, f"Box dim must be 5, got {bboxes_tensor.shape}"

        if keep.sum() == 0:
            empty_bboxes = RotatedBoxes(bboxes_tensor.new_zeros((0, 5)))
            gt_instances = InstanceData(
                bboxes=empty_bboxes,
                labels=pred.labels.new_zeros((0,), dtype=torch.long)
            )
        else:
            if max_num_per_img is not None and keep.sum() > max_num_per_img:
                scores = pred.scores[keep]
                _, topk = scores.topk(max_num_per_img)
                keep_idx = torch.where(keep)[0][topk]
                keep = torch.zeros_like(keep, dtype=torch.bool)
                keep[keep_idx] = True

            selected_bboxes = bboxes_tensor[keep].detach()
            rotated_bboxes = RotatedBoxes(selected_bboxes)
            gt_instances = InstanceData(
                bboxes=rotated_bboxes,
                labels=pred.labels[keep].detach()
            )

        new_ds = ds.clone()
        new_ds.gt_instances = gt_instances
        new_ds.r_gt_instances = gt_instances  # 兼容字段

        new_data_samples.append(new_ds)
    
    return new_data_samples

def get_alpha_exp(iter, max_iter, alpha_max=0.5, k=1.0):
    """
    指数增长（温和版）：alpha = alpha_max * (1 - exp(-k * iter / max_iter))
    k=1.0 时增长较平缓，大约在 iter ≈ max_iter 时才接近 alpha_max
    """
    return alpha_max * (1 - math.exp(-k * iter / max_iter))

def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='swint')
    parser.add_argument('--init_backbone', type=str, default='imp')
    parser.add_argument('--decoder', type=str, default='upernet')
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--save-path', type=str, required=True)
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--port', default=None, type=int)
    parser.add_argument('--load', type=str, default='backbone')
    parser.add_argument('--interval', type=int, default=1)
    parser.add_argument('--image_size', type=int, default=512)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    save_dir = Path(args.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Logger
    logger = init_log('global', logging.INFO)
    logger.propagate = False

    # DDP
    rank, world_size = setup_distributed(port=args.port)
    ddp = world_size > 1
    amp = cfg.get("amp", False)

    if rank == 0:
        logger.info("Args + Cfg:\n{}".format(pprint.pformat({**cfg, **vars(args)})))
        writer = SummaryWriter(save_dir)

    # Model
    cudnn.enabled, cudnn.benchmark = True, True
    # model = ORCNN(args, cfg).cuda()
    # repo_dir = '/data1/users/zhengzhiyu/ssl_workplace/dinov3-main'
    # backbone = torch.hub.load(repo_dir, 'dinov3_vitb16', source='local', weights='/data1/users/zhengzhiyu/ssl_workplace/UniMatch-V2-main/pretrained/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth')
    # model = ORCNN(args, cfg, backbone).cuda()
    model = ORCNN(args, cfg).cuda()

    if ddp:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = nn.parallel.DistributedDataParallel(model, device_ids=[int(os.environ["LOCAL_RANK"])],
                                                    broadcast_buffers=False, find_unused_parameters=True)
    if args.backbone in {'vit_l', 'vit_b', 'vit_h', 'vit_l_rvsa'}:
        model._set_static_graph()

    if rank == 0:
        logger.info(f"Total params: {count_params(model):.1f}M")

    # # Optimizer
    optim_cfg = dict(
        optimizer=dict(type='AdamW', lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01),
        paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                        'relative_position_bias_table': dict(decay_mult=0.),
                                        'norm': dict(decay_mult=0.)})
    )
    optimizer = build_optim_wrapper(model, optim_cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, cfg['total_iters'], eta_min=0)


    # optim_wrapper = dict(
    #     optimizer=dict(
    #     # type='AdamW', lr=0.00005, betas=(0.9, 0.999), weight_decay=0.05),
    #     type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=0.05),
    #     # constructor='LayerDecayOptimizerConstructor_ViT', 
    #     paramwise_cfg=dict(
    #         num_layers=12, 
    #         layer_decay_rate=0.9,
    #         )
    #         )  
    # param_scheduler = [
    #     dict(
    #         type='LinearLR',
    #         start_factor=1.0 / 3,
    #         by_epoch=False,
    #         begin=0,
    #         end=1000  # warmup 前 1000 iters
    #     ),
    #     dict(
    #         type='MultiStepLR',
    #         by_epoch=False,          # 👈 关键：改为按 iter
    #         begin=0,
    #         # end=args.end_iter,
    #         end=cfg['total_iters'],
    #         milestones=[80000, 110000],  # 按 iter 设置的衰减点
    #         gamma=0.1
    #     )
    # ]
    # optimizer = build_optim_wrapper(model, optim_wrapper)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, args.max_epoch*len(train_dataset), eta_min=0, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, cfg['total_iters'], eta_min=0)
    
    criterion_l = (nn.CrossEntropyLoss(**cfg['criterion']['kwargs'])
                   if cfg['criterion']['name'] == 'CELoss'
                   else ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs'])).cuda()
    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    train_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u', size=cfg['crop_size'], ignore_value=cfg['ignore_index'], id_path=args.unlabeled_id_path)

    train_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', size=cfg['crop_size'], id_path=args.labeled_id_path, nsample=len(train_u.ids))

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val', size=cfg['crop_size'])

    sup_loader_kwargs = dict(batch_size=cfg['batch_size'], pin_memory=True, drop_last=True, collate_fn=det_collate_fn, num_workers=cfg['workers'])
    unsup_loader_kwargs = dict(batch_size=cfg['batch_size'], pin_memory=True, drop_last=True, collate_fn=det_collate_fn_unsup, num_workers=cfg['workers'])
    if ddp:
        trainloader_l = DataLoader(train_l, sampler=torch.utils.data.distributed.DistributedSampler(train_l), **sup_loader_kwargs)
        trainloader_u = DataLoader(train_u, sampler=torch.utils.data.distributed.DistributedSampler(train_u), **unsup_loader_kwargs)
        valloader = DataLoader(valset, sampler=torch.utils.data.distributed.DistributedSampler(valset), 
        batch_size=8, pin_memory=True, collate_fn=det_collate_fn, num_workers=cfg['workers'])
        # batch_size=8, pin_memory=True, collate_fn=det_collate_fn, num_workers=cfg['workers'])
    else:
        trainloader_l = DataLoader(train_l, shuffle=True, **sup_loader_kwargs)
        valloader = DataLoader(valset, batch_size=4, shuffle=False, num_workers=1)
    # scheduler = build_param_scheduler(optimizer, param_scheduler, trainloader_l)
    # Training state
    iters, epoch, previous_best = -1, -1, 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    conf_thresh = cfg['conf_thresh']
    total_epochs = int(cfg['total_iters'] / len(trainloader_l)) + 1

    if os.path.exists(os.path.join(args.save_path, 'best_dinov3_vit_b_multi_28k.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'best_dinov3_vit_b_multi_28k.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        # epoch = checkpoint['epoch'] - 1
        iters = checkpoint['iters']

    for epoch in range(epoch + 1, total_epochs):
        if ddp:
            trainloader_l.sampler.set_epoch(epoch)
            trainloader_u.sampler.set_epoch(epoch)

        model.train()
        log_avg = DictAverageMeter()
        
        for i, ((img_x, mask_x, target),
                (img_u_w, img_u_s, ignore_mask, target_u)) in enumerate(zip(trainloader_l, trainloader_u)):
            
            t0 = time.time()
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s, ignore_mask = img_u_w.cuda(), img_u_s.cuda(), ignore_mask.cuda()
            # target: List[Dict] → DetDataSample
            data_samples = build_data_samples(img_x, target)
            data_samples_u = build_data_samples(img_u_w, target_u)
            
 
            with torch.no_grad():
                model.eval()
                pseudo_ds, u_w_out = model(img_u_w, data_samples_u)
                pseudo_gt_samples = extract_pseudo_labels_for_training(pseudo_ds['output_rd'], score_thr=0.7, max_num_per_img=100)
                conf_u_w, mask_u_w = u_w_out.softmax(dim=1).max(dim=1)

            model.train()
            with torch.cuda.amp.autocast(enabled=amp):
                loss_dict, x_out = model(img_x, data_samples, mask_ratio=cfg['mask_ratio'])   # ★ loss 在模型内部
                loss_u_dict, u_s_out = model(img_u_s, pseudo_gt_samples, mask_ratio=cfg['mask_ratio'])
                # loss_total = sum_loss_dict(loss_dict)
                loss_rd = parse_losses(loss_dict['loss_rd'])
                loss_rd_u = parse_losses(loss_u_dict['loss_rd'])
                loss_ss = criterion_l(x_out, mask_x)
                loss_ss_u = criterion_u(u_s_out, mask_u_w)
                loss_ss_u = confidence_weighted_loss(loss_ss_u, conf_u_w, ignore_mask, conf_thresh=conf_thresh)
                pse_mask_ratio = ((conf_u_w >= conf_thresh) & (ignore_mask != cfg['ignore_index'])).sum().item() / (ignore_mask != cfg['ignore_index']).sum()
                

                alpha = get_alpha_exp(iters, 20000, alpha_max=0.5, k=1.0)
                total_loss = (loss_rd + alpha * loss_rd_u + loss_ss + 0.5 * loss_ss_u) / 3.0
            if ddp:
                torch.distributed.barrier()
            optimizer.zero_grad()

            if amp:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

            log_info = {
                "loss_total": total_loss.detach().item(),
                "loss_rd_sup": loss_rd.detach().item(),      # supervised loss
                "loss_rd_unsup": alpha * loss_rd_u.detach().item(),  # unsupervised (pseudo-label) loss
                "loss_ss_sup": loss_ss.detach().item(),      # supervised loss
                "loss_ss_unsup": 0.5 * loss_ss_u.detach().item(),  # unsupervised (pseudo-label) loss
                "iter_time": time.time() - t0,
                "mask_ratio": pse_mask_ratio
            }

            log_avg.update(log_info)

            iters += 1
            scheduler.step()
            # scheduler_after_train_iter(scheduler)


            if rank == 0 and iters % cfg['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info("=====> LR: {:.5f}, Iteration: {:}/{:}, Epoch: {:}/{:}, log_avg: {}".format( 
                lr, iters, cfg['total_iters'], epoch, total_epochs, str(log_avg)))

            if (iters) % cfg['validate_interval'] == 0:
                if rank == 0:
                    logger.info('>>>>>>>>>>>>>>>> Start Evaluation of Finetune >>>>>>>>>>>>>>>>')

                start_time = time.time()
                primary_metric, _ = validation(args, cfg,logger, iters, model, valloader)
                end_time = time.time()
                is_best = primary_metric > previous_best
                previous_best = max(primary_metric, previous_best)
                if rank == 0 and is_best:
                    state = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "iters": iters,
                             "previous_best": previous_best}
                    backbone = {"model": model.module.encoder.state_dict()}
                    save_checkpoint(state, backbone, save_dir, args.backbone, "best", is_best)

            if iters % cfg['save_interval'] == 0 and rank == 0:
                # state = {"model": model.state_dict()}
                state = {"model": model.state_dict(),
                             "optimizer": optimizer.state_dict(),
                             "iters": iters,
                             "epoch": epoch}
                backbone = {"model": model.module.encoder.state_dict()}
                save_checkpoint(state, backbone, save_dir, args.backbone, "iter")


def save_checkpoint(state, backbone, save_dir: Path, backbone_name: str, suffix: str, is_best=False):
    """统一保存 checkpoint."""
    if is_best:
        torch.save(state, save_dir / f'best_{backbone_name}_obb_{suffix}.pth')
        torch.save(backbone, save_dir / f'best_{backbone_name}_{suffix}.pth')
    else:
        torch.save(state, save_dir / f'{backbone_name}_obb_iter_{state["iters"]}.pth')
        torch.save(backbone, save_dir / f'{backbone_name}_iter_{state["iters"]}.pth')



@torch.no_grad()
def validation(args, cfg, logger, iters, model, val_loader):
    model.eval()

    # === Detection Metric ===
    from model.ordet.rotated_detection.metric import MTP_RD_Metric
    rd_metric = MTP_RD_Metric(metric='mAP', predict_box_type='rbox')
    rd_metric.dataset_meta = val_loader.dataset.metainfo

    # === Segmentation Meters ===
    seg_meters = {k: AverageMeter() for k in ["intersection", "union", "target", "predict"]}

    eval_length = 0

    if dist.get_rank() == 0:
        pbar = tqdm(val_loader, desc="Validation")
    else:
        pbar = val_loader

    with torch.no_grad():
        for imgs, seg_targets, targets in pbar:
            imgs = imgs.cuda(non_blocking=True)
            seg_targets = seg_targets.long().cuda(non_blocking=True)

            # 构建 GT data_samples for detection
            data_samples = build_data_samples(imgs, targets)

            # Forward pass (multi-task)
            outputs, seg_logits = model(imgs, data_samples)  # dict: {'output_rd', 'output_seg'}

            # ====== Detection Evaluation ======
            rd_metric.process(
                data_batch=None,
                data_samples=outputs['output_rd']
            )

            # ====== Segmentation Evaluation ======

            if getattr(args, 'eval_mode', 'resize') == 'resize':
                ori_h, ori_w = seg_targets.shape[-2:]
                # Resize logits to original size
                seg_pred = F.interpolate(seg_logits, size=(ori_h, ori_w), mode='bilinear', align_corners=True)
                seg_pred = seg_pred.argmax(dim=1)  # [B, H, W]
            else:
                seg_pred = seg_logits.argmax(dim=1)

            # Compute IoU on GPU
            inter, union, tgt, prd = intersectionAndUnionGPU(
                seg_pred, seg_targets, 
                cfg['nclass'], 
                cfg['ignore_index'])

            # All-reduce across GPUs
            dist.all_reduce(inter); dist.all_reduce(union)
            dist.all_reduce(tgt);   dist.all_reduce(prd)

            for k, v in zip(seg_meters.keys(), [inter, union, tgt, prd]):
                seg_meters[k].update(v)

            eval_length += imgs.size(0)

    # ===== Finalize Detection Metrics =====
    rd_res = rd_metric.evaluate(eval_length)
    rd_map = rd_res.get('dota/mAP', 0.0) * 100

    # ===== Finalize Segmentation Metrics =====
    iou = seg_meters["intersection"].sum.cpu().numpy() / (seg_meters["union"].sum.cpu().numpy() + 1e-10) * 100
    acc = seg_meters["intersection"].sum.cpu().numpy() / (seg_meters["target"].sum.cpu().numpy() + 1e-10) * 100
    prec = seg_meters["intersection"].sum.cpu().numpy() / (seg_meters["predict"].sum.cpu().numpy() + 1e-10) * 100
    f1 = 2 * (prec * acc) / (prec + acc + 1e-10)

    # Skip background if needed (e.g., pretrain dataset)
    # if getattr(args, 'dataset', '') == 'pretrain_mt':
    #     valid_idx = slice(0, -1)  # or slice(1, None) based on your convention
    # else:
    valid_idx = slice(None)

    seg_miou = np.mean(iou[valid_idx])
    seg_macc = np.mean(acc[valid_idx])
    seg_mf1 = np.mean(f1[valid_idx])
    seg_all_acc = seg_meters["intersection"].sum[valid_idx].sum() / (seg_meters["target"].sum[valid_idx].sum() + 1e-10)

    primary_metric = (rd_map + seg_miou) / 2.0  # or just rd_map if detection is main task
    # ===== Logging =====
    if dist.get_rank() == 0:
        # --- 打印每个类别的 IoU ---
        # 假设 args.dataset_name 或 args.dataset 对应 CLASSES 的 key，例如 'isaid'
        dataset_name = getattr(args, 'dataset', 'pretrain_mt')  # 可根据实际字段调整
        if dataset_name in CLASSES:
            class_names = CLASSES[dataset_name]
        else:
            # 如果没有对应名称，用默认索引
            class_names = [f"Class_{i}" for i in range(len(iou))]

        logger.info("===== Per-class Segmentation IoU =====")
        for cls_idx, (name, iou_val) in enumerate(zip(class_names, iou)):
            logger.info(f'Class [{cls_idx:2d}] {name:<20} : {iou_val:.2f}%')

        logger.info(f'[Validation] iteration {iters}:')
        logger.info(f'Mean of mAP and mIoU = {primary_metric:.4f}')
        logger.info(f'RD rbox mAP = {rd_map:.4f}')
        logger.info(f'Seg mIoU = {seg_miou:.4f}, mAcc = {seg_macc:.4f}, mF1 = {seg_mf1:.4f}')


    # Return main metric (e.g., for saving best model) + all metrics
    acc_list = {
        'rd': rd_map,
        'seg_miou': seg_miou,
        'seg_macc': seg_macc,
        'seg_mf1': seg_mf1,
        'seg_all_acc': seg_all_acc,
        'iou_class': iou.tolist()
    }
 
    # You can choose which metric to return as primary (e.g., average of two)
    
    return primary_metric, acc_list

if __name__ == '__main__':
    set_seeds()
    main()
