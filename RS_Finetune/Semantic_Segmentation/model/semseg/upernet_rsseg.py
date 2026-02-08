import torch
import torch.nn as nn
from model.backbone.vit_win_rvsa_v3_wsz7_mtp import vit_b_rvsa, vit_l_rvsa
# from model.backbone.intern_image import InternImage
from model.backbone.vit import ViT_B, ViT_L, ViT_H
from model.backbone.vit_moe import ViT_B_MOE, ViT_L_MOE, ViT_H_MOE
# from model.backbone.vitaev2 import vitae_v2_s
from model.semseg.encoder_decoder import MTP_SS_UperNet
# from model.backbone.our_resnet import res50
from model.backbone.swin_transformer import swin_t
from model.backbone.swin import swin
from model.backbone.biformer.R3BiFormer import biformer_tiny, biformer_small
# from model.backbone.swin_transformer2 import SwinTransformer
import torch.nn.functional as F
from copy import deepcopy

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def initialize_head(module):
    for m in module.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc

def get_backbone(args):
    
    if args.backbone == 'swin_t':
        # encoder = SwinTransformer(depths=[2, 2, 6, 2],
        #             num_heads=[3, 6, 12, 24],
        #             window_size=7,
        #             ape=False,
        #             drop_path_rate=0.3,
        #             patch_norm=True
        #             )
        
        # encoder = swin_t()
        # encoder = swin_t()
        encoder = swin(embed_dim=96, 
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=7,
                    ape=False,
                    drop_path_rate=0.3,
                    patch_norm=True
                    )
        print('################# Using Swin-T as backbone! ###################')
        if args.init_backbone == 'rsp':
            encoder.init_weights('./pretrained/rsp-swin-t-ckpt.pth')
        if args.init_backbone == 'imp':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/ssl_pretrain/pretrained/swin_tiny_patch4_window7_224.pth')
            print('################# Initing Swin-T pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure Swin-T Pretraining! ###################')
        else:
            raise NotImplementedError
    
    if args.backbone == 'swin_b':
        encoder = swin(embed_dim=128, 
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
            ape=False,
            drop_path_rate=0.3,
            patch_norm=True
                    )
        print('################# Using Swin-B as backbone! ###################')
        if args.init_backbone == 'gfm':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/gfm.pth')
            print('################# Initing Swin-B pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure Swin-B Pretraining! ###################')
        else:
            raise NotImplementedError
        
    if args.backbone == 'swin_l':
        encoder = swin(embed_dim=192, 
                        depths=[2, 2, 18, 2],
                        num_heads=[6, 12, 24, 48],
                        window_size=7,
                        ape=False,
                        drop_path_rate=0.3,
                        patch_norm=True
                        )
        print('################# Using Swin-L as backbone! ###################')
        if args.init_backbone == 'imp':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/swin_large_patch4_window7_224_22k.pth')
            print('################# Initing Swin-T pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure Swin-T Pretraining! ###################')
        else:
            raise NotImplementedError


    if args.backbone == 'vit_b_rvsa':
        encoder = vit_b_rvsa(args)
        print('################# Using ViT-B + RVSA as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('/data1/users/zhengzhiyu/mtp_workplace/obb_mtp/pretrained/vit-b-checkpoint-1599.pth')
            print('################# Initing ViT-B + RVSA pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 's5':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/semi_sep/vit_b/best_vit_b_rvsa_ins.pth')
            print('################# Initing ViT-B S5 pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-B + RVSA SEP Pretraining! ###################')
        else:
            raise NotImplementedError

    elif args.backbone == 'vit_l_rvsa':
        encoder = vit_l_rvsa(args)
        print('################# Using ViT-L + RVSA as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('./pretrained/vit-l-mae-checkpoint-1599.pth')
            print('################# Initing ViT-L + RVSA pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'mae_mtp':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/last_vit_l_rvsa_ss_is_rd_pretrn_model_encoder.pth')
            print('################# Initing ViT-L + RVSA MTP pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 's5':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/semi_sep/vit_l/best_vit_l_rvsa_ins.pth')
            print('################# Initing ViT-L + RVSA S5 pretrained weights for Pretraining')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-L + RVSA SEP Pretraining! ###################')
        else:
            raise NotImplementedError


    elif args.backbone == 'vit_h':
        encoder = ViT_H(args)
        print('################# Using ViT-H as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/semi_sep/vit_h/vit-h-mae-checkpoint-1599.pth')
            print('################# Initing ViT-H pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-H SEP Pretraining! ###################')
        else:
            raise NotImplementedError


    elif args.backbone == 'vit_h_moe':
        encoder = ViT_H_MOE(args)
        print('################# Using ViT-H as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/semi_sep/vit_h/vit-h-mae-checkpoint-1599.pth')
            print('################# Initing ViT-H pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-H SEP Pretraining! ###################')
        else:
            raise NotImplementedError


    elif args.backbone == 'vit_l':
        encoder = ViT_L(args)
        print('################# Using ViT-L as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/vit-l-mae-checkpoint-1599.pth')
            print('################# Initing ViT-L pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'scale_mae':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/scalemae-vitlarge-800.pth')

        elif args.init_backbone == 'satmae_pp':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/ViT-L_satmae_pp_pretrain_fmow_rgb.pth')
            print('################# Initing ViT-L satmae_pp pretrained weights for Pretraining! ###################')
        
        elif args.init_backbone == 'none':
            print('################# Pure ViT-L SEP Pretraining! ###################')
        else:
            raise NotImplementedError


    elif args.backbone == 'vit_l_moe':
        encoder = ViT_L_MOE(args)
        print('################# Using ViT-L-MOE as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/vit-l-mae-checkpoint-1599.pth')
            print('################# Initing ViT-L-MOE pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'scale_mae':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/scalemae-vitlarge-800.pth')

        elif args.init_backbone == 'satmae_pp':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/ViT-L_satmae_pp_pretrain_fmow_rgb.pth')
            print('################# Initing ViT-L-MOE satmae_pp pretrained weights for Pretraining! ###################')
        
        elif args.init_backbone == 'none':
            print('################# Pure ViT-L-MOE SEP Pretraining! ###################')
        else:
            raise NotImplementedError


    elif args.backbone == 'vit_b':
        encoder = ViT_B(args)
        print('################# Using ViT-B as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('/data1/users/zhengzhiyu/mtp_workplace/obb_mtp/pretrained/vit-b-checkpoint-1599.pth')
            print('################# Initing ViT-B  pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'my_mae':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/Remote-Sensing-RVSA-main/Remote-Sensing-RVSA-main/mae-main/output/millionAID_224/1600_0.75_0.00015_0.05_1792/checkpoint-0.pth')
            print('################# Initing ViT-B My MAE pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'mae_imagenet':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep2/pretrained/mae_pretrain_vit_base.pth')
            print('################# Initing ViT-B MAE pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 's5':
            encoder.init_weights('/data1/users/zhengzhiyu/mtp_workplace/obb_mtp/pretrained/best_vit_b_ins.pth')
            print('################# Initing ViT-B S5 pretrained weights for Pretraining! ###################')

        elif args.init_backbone == 'none':
            print('################# Pure ViT-B SEP Pretraining! ###################')
        else:
            raise NotImplementedError

    elif args.backbone == 'vit_b_moe':
        encoder = ViT_B_MOE(args)
        print('################# Using ViT-B as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('/data1/users/zhengzhiyu/mtp_workplace/obb_mtp/pretrained/vit-b-checkpoint-1599.pth')
            print('################# Initing ViT-B  pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'my_mae':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/Remote-Sensing-RVSA-main/Remote-Sensing-RVSA-main/mae-main/output/millionAID_224/1600_0.75_0.00015_0.05_1792/checkpoint-0.pth')
            print('################# Initing ViT-B My MAE pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'mae_imagenet':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep2/pretrained/mae_pretrain_vit_base.pth')
            print('################# Initing ViT-B MAE pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 's5':
            encoder.init_weights('/data1/users/zhengzhiyu/mtp_workplace/obb_mtp/pretrained/best_vit_b_ins.pth')
            print('################# Initing ViT-B S5 pretrained weights for Pretraining! ###################')

        elif args.init_backbone == 'none':
            print('################# Pure ViT-B SEP Pretraining! ###################')
        else:
            raise NotImplementedError

    elif args.backbone == 'R3B_S':
        encoder = biformer_small(args)
        print('################# Using R3BiFormer-S as backbone! ###################')
        if args.init_backbone == 'imp':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/biformer_small_best.pth')
            print('################# Initing R3BiFormer-S  pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure R3BiFormer-S SEP Pretraining! ###################')
        else:
            raise NotImplementedError




    elif args.backbone == 'internimage_xl':
        encoder = InternImage(core_op='DCNv3',
                        channels=192,
                        depths=[5, 5, 24, 5],
                        groups=[12, 24, 48, 96],
                        mlp_ratio=4.,
                        drop_path_rate=0.2,
                        norm_layer='LN',
                        layer_scale=1e-5,
                        offset_scale=2.0,
                        post_norm=True,
                        with_cp=True,
                        out_indices=(0, 1, 2, 3)
                        )
        print('################# Using InternImage-XL as backbone! ###################')
        if args.init_backbone == 'imp':
            encoder.init_weights('./pretrained/internimage_xl_22kto1k_384.pth')
            print('################# Initing InterImage-T pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure InterImage-T SEP Pretraining! ###################')
        else:
            raise NotImplementedError
        
    elif args.backbone == 'vitaev2_s':
        print('################# Using ViTAEv2-S as backbone! ###################')
        encoder = vitae_v2_s(args)
        if args.init_backbone == 'rsp':
            encoder.init_weights("./pretrained/rsp-vitaev2-s-ckpt.pth")
            print('################# Using RSP as pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure ViTAEV2-S Pretraining! ###################')
        else:
            raise NotImplementedError
        
    elif args.backbone == 'resnet50':
        print('################# Using ResNet-50 as backbone! ###################')
        encoder = res50()
        if args.init_backbone == 'rsp':
            encoder.init_weights("./pretrained/rsp-resnet-50-ckpt.pth")
            print('################# Using RSP as pretraining! ###################')
        elif args.init_backbone == 'imp':
            encoder.init_weights("./pretrained/resnet50-0676ba61.pth")
        elif args.init_backbone == 'none':
            print('################# Pure  Pretraining! ###################')
        else:
            raise NotImplementedError



    return encoder

def get_semsegdecoder(in_channels):
    semsegdecoder = MTP_SS_UperNet(
    decode_head = dict(
                type='UPerHead',
                num_classes = 1,
                in_channels=in_channels,
                ignore_index=255,
                in_index=[0, 1, 2, 3],
                pool_scales=(1, 2, 3, 6),
                channels=256,
                dropout_ratio=0.1,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                align_corners=False,
                loss_decode=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)
                ))
    return semsegdecoder


# class UperNet(torch.nn.Module):
#     def __init__(self, args, cfg):
#         super(UperNet, self).__init__()

#         self.args = args
#         self.encoder = get_backbone(args)
#         # Init task head
#         print('################# Using UperNet for semseg! ######################')
#         self.semsegdecoder = get_semsegdecoder(in_channels=getattr(self.encoder, 'out_channels', None))

#         self.semseg_heads = nn.ModuleDict({
#             'vaihingen': nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, cfg['vaihingen']['nclass'], 1)),
#             'potsdam': nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, cfg['potsdam']['nclass'], 1)),
#             'OpenEarthMap': nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, cfg['OpenEarthMap']['nclass'], 1)),
#             'loveda': nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, cfg['loveda']['nclass'], 1)),
#             'UDD': nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, cfg['UDD']['nclass'], 1)),
#             'VDD': nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, cfg['VDD']['nclass'], 1)),
#         })


#     def forward(self, x, dataset='vaihingen'):
#         b, c, h, w = x.shape
#         if self.args.backbone == 'vit_b_moe':
#             e = self.encoder(x, dataset)
#         else:
#             e = self.encoder(x)
#         ss = self.semsegdecoder.decode_head._forward_feature(e)
#         if dataset not in self.semseg_heads:
#             raise ValueError(f"Unknown dataset name: {dataset}")
#         out = self.semseg_heads[dataset](ss)
#         out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
#         return out



class UperNet(torch.nn.Module):
    def __init__(self, args, cfg):
        super(UperNet, self).__init__()

        self.args = args
        self.encoder = get_backbone(args)
        print('################# Using UperNet for semseg! ######################')

        # 用于加载预训练参数的主解码器
        self.semsegdecoder = get_semsegdecoder(in_channels=getattr(self.encoder, 'out_channels', None))

        # 为每个数据集复制 decoder 和 head
        self.semsegdecoder_potsdam = deepcopy(self.semsegdecoder)
        self.semsegdecoder_OpenEarthMap = deepcopy(self.semsegdecoder)
        self.semsegdecoder_loveda = deepcopy(self.semsegdecoder)
        self.semsegdecoder_UDD = deepcopy(self.semsegdecoder)
        self.semsegdecoder_uavid = deepcopy(self.semsegdecoder)

        # 定义每个数据集的 head（只做分类）
        self.semseghead_vaihingen = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(256, cfg['vaihingen']['nclass'], 1)
        )
        self.semseghead_potsdam = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(256, cfg['potsdam']['nclass'], 1)
        )
        self.semseghead_OpenEarthMap = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(256, cfg['OpenEarthMap']['nclass'], 1)
        )
        self.semseghead_loveda = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(256, cfg['loveda']['nclass'], 1)
        )
        self.semseghead_UDD = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(256, cfg['UDD']['nclass'], 1)
        )
        self.semseghead_uavid = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(256, cfg['uavid']['nclass'], 1)
        )

    def forward(self, x, dataset='vaihingen'):
        b, c, h, w = x.shape

        if self.args.backbone == 'vit_b_moe':
            e = self.encoder(x, dataset)
        else:
            e = self.encoder(x)

        if dataset == 'vaihingen':
            decoder = self.semsegdecoder
            head = self.semseghead_vaihingen
        elif dataset == 'potsdam':
            decoder = self.semsegdecoder_potsdam
            head = self.semseghead_potsdam
        elif dataset == 'OpenEarthMap':
            decoder = self.semsegdecoder_OpenEarthMap
            head = self.semseghead_OpenEarthMap
        elif dataset == 'loveda':
            decoder = self.semsegdecoder_loveda
            head = self.semseghead_loveda
        elif dataset == 'UDD':
            decoder = self.semsegdecoder_UDD
            head = self.semseghead_UDD
        elif dataset == 'uavid':
            decoder = self.semsegdecoder_uavid
            head = self.semseghead_uavid
        else:
            raise ValueError(f"Unknown dataset name: {dataset}")

        # decoder 只负责提取语义特征
        ss = decoder.decode_head._forward_feature(e)

        # head 负责分类
        out = head(ss)

        # 恢复到原图大小
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out



class UperNet(torch.nn.Module):
    def __init__(self, args, cfg):
        super(UperNet, self).__init__()

        self.args = args
        self.encoder = get_backbone(args)
        print('################# Using UperNet for semseg! ######################')

        # 用于加载预训练参数的主解码器
        self.semsegdecoder = get_semsegdecoder(in_channels=getattr(self.encoder, 'out_channels', None))

        # 为每个数据集复制 decoder 和 head
        self.semsegdecoder_potsdam = deepcopy(self.semsegdecoder)
        self.semsegdecoder_OpenEarthMap = deepcopy(self.semsegdecoder)
        self.semsegdecoder_loveda = deepcopy(self.semsegdecoder)
        # self.semsegdecoder_UDD = deepcopy(self.semsegdecoder)
        # self.semsegdecoder_VDD = deepcopy(self.semsegdecoder)

        # 定义每个数据集的 head（只做分类）
        self.semseghead_vaihingen = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(256, cfg['vaihingen']['nclass'], 1)
        )
        self.semseghead_potsdam = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(256, cfg['potsdam']['nclass'], 1)
        )
        self.semseghead_OpenEarthMap = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(256, cfg['OpenEarthMap']['nclass'], 1)
        )
        self.semseghead_loveda = nn.Sequential(
            nn.Dropout2d(0.1), nn.Conv2d(256, cfg['loveda']['nclass'], 1)
        )
        # self.semseghead_UDD = nn.Sequential(
        #     nn.Dropout2d(0.1), nn.Conv2d(256, cfg['UDD']['nclass'], 1)
        # )
        # self.semseghead_VDD = nn.Sequential(
        #     nn.Dropout2d(0.1), nn.Conv2d(256, cfg['VDD']['nclass'], 1)
        # )

    def forward(self, x, dataset='vaihingen'):
        b, c, h, w = x.shape

        if self.args.backbone == 'vit_b_moe' or self.args.backbone == 'vit_l_moe' or self.args.backbone == 'vit_h_moe':
            e = self.encoder(x, dataset)
        else:
            e = self.encoder(x)

        if dataset == 'vaihingen':
            decoder = self.semsegdecoder
            head = self.semseghead_vaihingen
        elif dataset == 'potsdam':
            decoder = self.semsegdecoder_potsdam
            head = self.semseghead_potsdam
        elif dataset == 'OpenEarthMap':
            decoder = self.semsegdecoder_OpenEarthMap
            head = self.semseghead_OpenEarthMap
        elif dataset == 'loveda':
            decoder = self.semsegdecoder_loveda
            head = self.semseghead_loveda
        # elif dataset == 'UDD':
        #     decoder = self.semsegdecoder_UDD
        #     head = self.semseghead_UDD
        # elif dataset == 'VDD':
        #     decoder = self.semsegdecoder_VDD
        #     head = self.semseghead_VDD
        else:
            raise ValueError(f"Unknown dataset name: {dataset}")

        # decoder 只负责提取语义特征
        ss = decoder.decode_head._forward_feature(e)

        # head 负责分类
        out = head(ss)

        # 恢复到原图大小
        out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        return out


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        x1=self.fn(x)
        return x1+x


if __name__ =="__main__":
    # MutliTaskPretrnFramework()
    model = UperNet(args).cuda()
    class Args:
        def __init__(self):
            self.backbone = 'swin_t'  # Backbone selection
            self.init_backbone = 'none'  # Pretraining method for backbone
            self.terr_nclass = 6  # Number of terrain classes for segmentation
            self.ins_nclass = 5  # Number of instance classes for segmentation

    # 实例化配置参数
    args = Args()

    # 创建 UperNet 模型实例
    model = UperNet(args)

    # 将模型设置为评估模式
    model.eval()

    # 生成一个随机输入 (假设输入尺寸为 [batch_size, channels, height, width])
    input_tensor = torch.randn(1, 3, 224, 224)  # 创建一个随机输入张量

    # 将输入传递给模型并获得输出
    with torch.no_grad():
        output = model(input_tensor)

    # 打印输出的形状
    print(output.shape)









