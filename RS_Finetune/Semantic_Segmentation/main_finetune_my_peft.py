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
# from dataset.finetune import SemiDataset, ValDataset
from dataset.finetune_sar import SemiDataset, ValDataset
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, init_log, intersectionAndUnion, intersectionAndUnionGPU
from util.dist_helper import setup_distributed
from model.semseg.upernet import UperNet
import torch.nn.functional as F
# from model.semseg.deeplabv3plus_vit import DeepLabV3Plus
from peft.tuners.semift import SemiFTConfig, AdaptModel

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
parser.add_argument('--resume', type=str, default='/data1/users/lvliang/project_123/S5_finetune/pretrained/best_dinov3_vit_b_SemiM3P_48k.pth', help='resume name')



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
    if cfg['dataset'] == 'isaid_ori':
        mIoU = np.nanmean(iou_class[1:]) * 100.0
        mAcc = np.nanmean(accuracy_class[1:]) * 100.0
        mF1 = np.nanmean(F1_class[1:]) * 100.0
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    else:
        mIoU = np.nanmean(iou_class) * 100.0
        mAcc = np.nanmean(accuracy_class) * 100.0
        mF1 = np.nanmean(F1_class) * 100.0
        allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    return mIoU, mAcc, mF1, allAcc, iou_class, F1_class


def apply_peft(model, cfg):
    peft_config = SemiFTConfig(
        method=cfg['peft'],
        target_modules=cfg['target_modules'],
        modules_to_save='head',
        bias='lora_only',
        nclass = cfg['nclass'],
    )
    model = AdaptModel(peft_config, model)
    return model


def show_trainable_parameters(model, logger):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = 0
    trainable_params_names = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params += param.numel()
            trainable_params_names.append(name)
    
    percentage = 100 * trainable_params / total_params if total_params > 0 else 0
    
    logger.info('--- 模型可训练参数 ---')
    logger.info('--- 可训练模块/参数列表---')

    for name in trainable_params_names:
        logger.info(f' - {name}')
        
    logger.info("\n --- 统计信息 ---")
    logger.info(f" - 总参数数量: {total_params:,}")
    logger.info(f" - 可训练参数数量: {trainable_params:,}")
    logger.info(f" - 可训练参数占比: {percentage:.2f}%")


def sync_adapter_from_ffn(model, copy_weights=True, zero_init_out=True, verbose=True):
    """
    For segmentation model where ViT is under model.encoder.blocks.
    """
    real_model = model.module if hasattr(model, "module") else model

    if not hasattr(real_model, "encoder") or not hasattr(real_model.encoder, "blocks"):
        if verbose:
            print("[sync_adapter_from_ffn] No encoder.blocks found. Skip.")
        return 0

    synced = 0
    for i, blk in enumerate(real_model.encoder.blocks):
        # blk should be SelfAttentionBlock
        if not hasattr(blk, "adapter_ffn") or blk.adapter_ffn is None:
            continue
        if not hasattr(blk, "mlp") or blk.mlp is None:
            continue

        ffn = blk.mlp
        adp = blk.adapter_ffn

        # ---- MLP case: fc1/fc2 ----
        if hasattr(ffn, "fc1") and hasattr(ffn, "fc2") and hasattr(adp, "fc1") and hasattr(adp, "fc2"):
            if copy_weights:
                with torch.no_grad():
                    adp.fc1.weight.copy_(ffn.fc1.weight)
                    if ffn.fc1.bias is not None and adp.fc1.bias is not None:
                        adp.fc1.bias.copy_(ffn.fc1.bias)
                    adp.fc2.weight.copy_(ffn.fc2.weight)
                    if ffn.fc2.bias is not None and adp.fc2.bias is not None:
                        adp.fc2.bias.copy_(ffn.fc2.bias)

            if zero_init_out:
                with torch.no_grad():
                    adp.fc2.weight.zero_()
                    if adp.fc2.bias is not None:
                        adp.fc2.bias.zero_()

            synced += 1
            continue

        # ---- SwiGLU case: w1/w2/w3 ----
        if (hasattr(ffn, "w1") and hasattr(ffn, "w2") and hasattr(ffn, "w3") and
            hasattr(adp, "w1") and hasattr(adp, "w2") and hasattr(adp, "w3")):
            if copy_weights:
                with torch.no_grad():
                    adp.w1.weight.copy_(ffn.w1.weight)
                    if ffn.w1.bias is not None and adp.w1.bias is not None:
                        adp.w1.bias.copy_(ffn.w1.bias)
                    adp.w2.weight.copy_(ffn.w2.weight)
                    if ffn.w2.bias is not None and adp.w2.bias is not None:
                        adp.w2.bias.copy_(ffn.w2.bias)
                    adp.w3.weight.copy_(ffn.w3.weight)
                    if ffn.w3.bias is not None and adp.w3.bias is not None:
                        adp.w3.bias.copy_(ffn.w3.bias)

            if zero_init_out:
                with torch.no_grad():
                    adp.w3.weight.zero_()
                    if adp.w3.bias is not None:
                        adp.w3.bias.zero_()

            synced += 1
            continue

    if verbose:
        print(f"[sync_adapter_from_ffn] synced adapters: {synced}")

    return synced


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    cfg['target_modules'] = ['mlp']

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
            modality = getattr(args, 'modality', 'opt')
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

    # 复制FFN的参数到外接的Adapter
    sync_adapter_from_ffn(model, copy_weights=True, zero_init_out=True, verbose=True)

    lr = { "vit_h": 0.00005, "vit_l": 0.00005, "vit_b": 0.00005, "dinov3_vit_b":0.00005,  "dinov3_vit_b_mae":0.00005, "vit_l_rvsa":0.00005 }.get(args.backbone, 0.00005)


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

    valloader = DataLoader(valset, batch_size=val_batch, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)

    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    scaler = torch.cuda.amp.GradScaler()
    amp = cfg['amp']
    mask_ratio = 0.0
    patch_size = 16
    # if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
    #     checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location=torch.device('cpu'))
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     epoch = checkpoint['epoch']
    #     previous_best = checkpoint['previous_best']
        
        # if rank == 0:
        #     logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    

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

                # B, C, H, W = img.shape
                # L = (H // patch_size) * (W // patch_size)
                # dummy_tokens = torch.empty(B, L, 1, device=img.device)
                # masks_bool, ids_keep = generate_masks(dummy_tokens, mask_ratio)

                pred = model(img)
                # pred = model(img, mask_ratio=cfg['mask_ratio'])
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
