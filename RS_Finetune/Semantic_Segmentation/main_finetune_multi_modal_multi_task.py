import argparse
import logging
import os
import pprint
import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD, AdamW, lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
import torch.distributed as dist
import numpy as np
import random
# from evaluate import evaluate
from dataset.finetune import SemiDataset, ValDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, init_log, intersectionAndUnion, intersectionAndUnionGPU
from util.dist_helper import setup_distributed
from model.semseg.upernet import UperNet
import torch.nn.functional as F
# from model.semseg.deeplabv3plus_vit import DeepLabV3Plus

parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--backbone', type=str, default='swint', required=True)
parser.add_argument('--init_backbone', type=str, default='imp', required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--image_size', type=int, default=512)
parser.add_argument('--interval', default=1, type=int, help='valid interval')
parser.add_argument('--load', type=str, default='none', choices=['backbone','network', 'none'], help='loaded model part')
# parser.add_argument('--resume', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/semi_sep/exp/pretrain/semi_sep_no_sam_refine/swint_upernet_imp/swint_upernet_semi_sep_iter_200000.pth', help='resume name')
# parser.add_argument('--resume', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/semi_sep/vit_l/rf_vit_l_upernet_semi_sep_iter_75000.pth', help='resume name')
# parser.add_argument('--resume', type=str, default='/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/semi_sep/vit_h/vit_h_upernet_miou_69_iter_30000.pth', help='resume name')
# parser.add_argument('--resume', type=str,  default='/data1/users/zhengzhiyu/mtp_workplace/mtpft_hwexp/exp_semi/vit_b/sdp/vit_b/rd_/interact_None/1.0_0.0_0.0/vit_b_s4p_upernet.pth', help='resume name')
parser.add_argument('--resume', type=str,  default='/data1/users/zhengzhiyu/ssl_workplace/S5_fulll/S4_Pretrain/exp/semi_mtm_mtp/dinov3_vit_b/labeled_mota_ms_merge_IRSAMap_lr5e_5/best_dinov3_vit_b_m3t_36k.pth', help='resume name')


def set_seeds(seed=2024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@torch.no_grad()
def validation_cpu(cfg, args, model, valid_loader):

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    predict_meter = AverageMeter()

    model.eval()

    for (x, y) in valid_loader:
        x = x.cuda()

        if cfg['eval_mode'] == 'slide_window':
            b, _, h, w = x.shape    # 获取输入图像的尺寸 (batch, channels, height, width)
            final = torch.zeros(b, cfg['nclass'], h, w).cuda()  # 用于存储最终预测结果
            size = cfg['crop_size']
            step = 510
            b = 0
            a = 0
            while (a <= int(h / step)):
                while (b <= int(w / step)):
                    sub_input = x[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)]
                    # print("sub_input.shape", sub_input.shape)
                    mask = model(sub_input) 
                    final[:,:, min(a * step, h - size): min(a * step + size, h), min(b * step, w - size):min(b * step + size, w)] += mask
                    b += 1
                b = 0
                a += 1
            o = final.argmax(dim=1)
        
        elif cfg['eval_mode'] == 'resize':
        # 使用缩放方式进行预测
            original_shape = x.shape[-2:]  # 保存原始图像的尺寸 (h, w)
            resized_x = F.interpolate(x, size=cfg['crop_size'], mode='bilinear', align_corners=True)
            resized_o = model(resized_x)   
            # 将预测结果复原到原始尺寸
            o = F.interpolate(resized_o, size=original_shape, mode='bilinear', align_corners=True)
            o = o.argmax(dim=1)

        else:
            # 直接进行预测（非滑动窗口模式）
            o = model(x)
            o = o.max(1)[1]
        gray = np.uint8(o.cpu().numpy())
        target = np.array(y, dtype=np.int32)
        intersection, union, target, predict = intersectionAndUnion(gray, target, cfg['nclass'], cfg['ignore_index'])
        reduced_intersection = torch.from_numpy(intersection).cuda()
        reduced_union = torch.from_numpy(union).cuda()
        reduced_target = torch.from_numpy(target).cuda()
        reduced_predict = torch.from_numpy(predict).cuda()

        dist.all_reduce(reduced_intersection)
        dist.all_reduce(reduced_union)
        dist.all_reduce(reduced_target)
        dist.all_reduce(reduced_predict)
        intersection_meter.update(reduced_intersection.cpu().numpy())
        union_meter.update(reduced_union.cpu().numpy())
        target_meter.update(reduced_target.cpu().numpy())
        predict_meter.update(reduced_predict.cpu().numpy())
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    precise_class = intersection_meter.sum / (predict_meter.sum + 1e-10)
    F1_class = 2*(precise_class*accuracy_class) / (precise_class+accuracy_class)
    
    mIoU = np.nanmean(iou_class) * 100.0
    mAcc = np.nanmean(accuracy_class) * 100.0
    mF1 = np.nanmean(F1_class) * 100.0
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc, iou_class, F1_class


import torch.nn.functional as F

def resize_pos_embed(pretrained_pos_embed, model_pos_embed):
    """
    将预训练 ViT 位置编码通过双线性插值调整为当前模型分辨率。
    pretrained_pos_embed: [1, L1, C]
    model_pos_embed:      [1, L2, C]
    """
    # 去掉 batch 维
    old_num_patches = pretrained_pos_embed.shape[1]
    new_num_patches = model_pos_embed.shape[1]

    if old_num_patches == new_num_patches:
        return pretrained_pos_embed

    # ViT-B: 通常没有 cls token，若有 cls token 需切分
    # 假设没有 cls_token
    C = pretrained_pos_embed.shape[-1]

    # 计算原始 grid 高宽
    old_size = int(old_num_patches ** 0.5)
    new_size = int(new_num_patches ** 0.5)

    pretrained_pos_embed = pretrained_pos_embed.reshape(1, old_size, old_size, C).permute(0, 3, 1, 2)  # [1,C,H,W]

    # 插值到新大小
    new_pos_embed = F.interpolate(
        pretrained_pos_embed,
        size=(new_size, new_size),
        mode='bicubic',
        align_corners=False
    )

    new_pos_embed = new_pos_embed.permute(0, 2, 3, 1).reshape(1, new_size * new_size, C)

    return new_pos_embed


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = UperNet(args, cfg)
    # model = DeepLabV3Plus(cfg, args)

    # for param in model.encoder.parameters():
    #     param.requires_grad = False

    # for name, module in model.named_children():
    #     if name != "semsegdecoder":
    #         for param in module.parameters():
    #             param.requires_grad = False


    lr = { "vit_h": 0.00005, "vit_l": 0.00005, "vit_b": 0.00005, "vit_b_rvsa":0.00005, "vit_l_rvsa":0.00005 }.get(args.backbone, 0.0001)

    if args.load == 'network':
        if os.path.isfile(args.resume):
            if rank == 0:
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location='cpu')
            if rank == 0:
                logger.info("=> loading ft model...")
            ckpt_dict = checkpoint['model']

            if list(ckpt_dict.keys())[0].startswith('module.'):
                ckpt_dict = {k[7:]: v for k, v in ckpt_dict.items()}

            model_dict = model.state_dict()

            # === 处理 pos_embed（保持不变）===
            if "backbone.pos_embed" in ckpt_dict and "backbone.pos_embed" in model_dict:
                if ckpt_dict["backbone.pos_embed"].shape != model_dict["backbone.pos_embed"].shape:
                    if rank == 0:
                        logger.warning(f"Resizing pos_embed from {ckpt_dict['backbone.pos_embed'].shape} "
                                    f"to {model_dict['backbone.pos_embed'].shape}")
                    ckpt_dict["backbone.pos_embed"] = resize_pos_embed(
                        ckpt_dict["backbone.pos_embed"],
                        model_dict["backbone.pos_embed"]
                    )

            # === 确定当前微调的模态 ===
            modality = getattr(args, 'modality', 'opt')  # 默认 'opt'
            if rank == 0:
                logger.info(f"Loading segmentation decoder/head for modality: {modality}")

            # === 构建新的 filtered_ckpt_dict，支持模态映射 ===
            filtered_ckpt_dict = {}

            for k, v in ckpt_dict.items():
                # 情况1：普通参数（如 encoder/backbone），直接匹配
                if k in model_dict and model_dict[k].shape == v.shape:
                    filtered_ckpt_dict[k] = v
                    continue

                # 情况2：来自 semsegdecoder.{modality} → 映射到 semsegdecoder
                if k.startswith(f'semsegdecoder.{modality}.'):
                    new_k = 'semsegdecoder.' + k[len(f'semsegdecoder.{modality}.'):]
                    if new_k in model_dict and model_dict[new_k].shape == v.shape:
                        filtered_ckpt_dict[new_k] = v
                        if rank == 0:
                            logger.info(f"Loaded {k} -> {new_k}")
                        continue

                # 情况3：来自 semseghead.{modality} → 映射到 semseghead
                if k.startswith(f'semseghead.{modality}.'):
                    new_k = 'semseghead.' + k[len(f'semseghead.{modality}.'):]
                    if new_k in model_dict and model_dict[new_k].shape == v.shape:
                        filtered_ckpt_dict[new_k] = v
                        if rank == 0:
                            logger.info(f"Loaded {k} -> {new_k}")
                        continue

                # 其他不匹配的参数跳过
                if rank == 0:
                    logger.warning(f"Skipping parameter: {k} with shape {v.shape} (does not match)")

            # 更新模型字典
            model_dict.update(filtered_ckpt_dict)
            model.load_state_dict(model_dict, strict=False)

    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))


    from mmengine.optim import build_optim_wrapper

    optim_wrapper = dict(
        optimizer=dict(
        type='AdamW', lr=lr, betas=(0.9, 0.999), weight_decay=0.05),
        paramwise_cfg=dict(
            num_layers=12, 
            layer_decay_rate=0.9,
            )
            )
    optimizer = build_optim_wrapper(model, optim_wrapper)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.optimizer, cfg['epochs'], eta_min=0, last_epoch=-1)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)

    if args.backbone == 'vit_l' or args.backbone == 'vit_b' or args.backbone == 'vit_h' or args.backbone == 'vit_l_rvsa':
        model._set_static_graph()

    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    trainset = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l', size=cfg['crop_size'], id_path=args.labeled_id_path)
    valset = ValDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset, num_replicas=world_size,rank=rank)
    trainloader = DataLoader(trainset, batch_size=cfg['batch_size'], shuffle=(trainsampler is None),
                             pin_memory=True, num_workers=1, drop_last=True, sampler=trainsampler)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset, num_replicas=world_size, rank=rank)
    
    val_batch = 1 if cfg['dataset'] == 'OpenEarthMap' else 8

    valloader = DataLoader(valset, batch_size=2, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    scaler = torch.cuda.amp.GradScaler()
    amp = cfg['amp']

    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        
        total_loss = AverageMeter()
        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):

            img, mask = img.cuda(), mask.cuda()

            with torch.cuda.amp.autocast(enabled=amp):
                model.train()
                pred = model(img)
                sup_loss = criterion(pred, mask)                
                torch.distributed.barrier()
                optimizer.zero_grad()
                loss = scaler.scale(sup_loss)
                loss.backward()
                scaler.step(optimizer)
                scaler.update()

            total_loss.update(sup_loss)
            iters = epoch * len(trainloader) + i

            if rank == 0:
                writer.add_scalar('train/loss_all', sup_loss.item(), iters)
                writer.add_scalar('train/loss_x', sup_loss.item(), iters)
            
            if (i % (max(2, len(trainloader) // 8)) == 0) and (rank == 0):
                # logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
                logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, sup_loss.item()))

        scheduler.step()

        if (epoch + 1) % args.interval == 0:
            start_time = time.time()
            mIoU, mAcc, mF1, allAcc, iou_class, F1_class = validation_cpu(cfg, args, model, valloader)
            end_time = time.time()

            if rank == 0:
                for (cls_idx, iou) in enumerate(iou_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                                'IoU: {:.4f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
                # logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
                logger.info('Last: validation epoch [{}/{}]: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}. Cost {:.4f} secs \n'.format(epoch+1, cfg['epochs'], mIoU, mAcc, mF1, allAcc, end_time-start_time))

                for (cls_idx, F1) in enumerate(F1_class):
                    logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
                                'F1 score: {:.4f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], F1))
                # logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
                logger.info('Last: validation epoch [{}/{}]: mIoU/mAcc/mF1/allAcc {:.4f}/{:.4f}/{:.4f}/{:.4f}. Cost {:.4f} secs'.format(epoch+1, cfg['epochs'], mIoU, mAcc, mF1, allAcc, end_time-start_time))
                    
                writer.add_scalar('eval/mIoU', mIoU, epoch)
                for i, iou in enumerate(iou_class):
                    writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)

            is_best = mIoU > previous_best
            previous_best = max(mIoU, previous_best)
            if rank == 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'epoch': epoch,
                    'previous_best': previous_best,
                }
                torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
                if is_best:
                    torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    set_seeds(1234)
    main()
