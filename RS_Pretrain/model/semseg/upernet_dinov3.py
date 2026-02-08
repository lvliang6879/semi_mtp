import torch
import torch.nn as nn
from model.backbone.vit_win_rvsa_v3_wsz7_mtp import vit_b_rvsa, vit_l_rvsa
# from model.backbone.intern_image import InternImage
from model.backbone.vit import ViT_B, ViT_L, ViT_H
# from model.backbone.vitaev2 import vitae_v2_s
from model.semseg.encoder_decoder import MTP_SS_UperNet
# from model.backbone.our_resnet import res50
from model.backbone.swin_transformer import swin_t
from model.backbone.swin import swin
from model.backbone.biformer.R3BiFormer import biformer_tiny, biformer_small
# from model.backbone.swin_transformer2 import SwinTransformer
import torch.nn.functional as F

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
            encoder.init_weights('./pretrained/swin_tiny_patch4_window7_224.pth')
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
            patch_norm=True,
            use_checkpoint=False
                    )
        print('################# Using Swin-T as backbone! ###################')
        if args.init_backbone == 'imp':
            encoder.init_weights('./pretrained/swin_base_patch4_window7_224_22k_20220317-4f79f7c0.pth')
            print('################# Initing Swin-T pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure Swin-T Pretraining! ###################')
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
            print('################# Pure ViT-L + RVSA SEP Pretraining! ###################')
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


    elif args.backbone == 'vit_g':
        encoder = ViT_G(args)
        print('################# Using ViT-G as backbone! ###################')
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
            encoder.init_weights('./pretrained/vit-l-mae-checkpoint-1599.pth')
            print('################# Initing ViT-L pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 'selectivemae':
            encoder.init_weights('/data1/users/zhengzhiyu/ssl_workplace/semi_sep/pretrained/semi_sep/vit_l/vit-large-opticalrs13m.pth')
            print('################# Initing ViT-L selectivemae Pretraining! ###################')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-L SEP Pretraining! ###################')
        else:
            raise NotImplementedError


    elif args.backbone == 'vit_b':
        encoder = ViT_B(args)
        print('################# Using ViT-B as backbone! ###################')
        if args.init_backbone == 'mae':
            encoder.init_weights('/data1/users/zhengzhiyu/mtp_workplace/obb_mtp/pretrained/vit-b-checkpoint-1599.pth')
            print('################# Initing ViT-B  pretrained weights for Pretraining! ###################')
        elif args.init_backbone == 's5':
            encoder.init_weights('/data1/users/zhengzhiyu/mtp_workplace/obb_mtp/pretrained/best_vit_b_ins.pth')
        elif args.init_backbone == 'none':
            print('################# Pure ViT-B SEP Pretraining! ###################')
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


class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    

class UperNet(torch.nn.Module):
    def __init__(self, args, cfg, backbone):
        super(UperNet, self).__init__()
        self.args = args
        self.encoder = backbone
        self.intermediate_layer_idx = {
            'small': [2, 5, 8, 11],
            'base': [2, 5, 8, 11],
        }

        print('################# Using UperNet for semseg! ######################')
        self.semsegdecoder = get_semsegdecoder(in_channels=[768, 768, 768, 768])

        # FPN modules
        self.fpn_modules = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2),
                Norm2d(768),
                nn.GELU(),
                nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(768, 768, kernel_size=2, stride=2),
            ),
            nn.Identity(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        ])

        # Semseg head
        self.semseghead = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(256, cfg['nclass'], kernel_size=1)
        )

    def forward(self, x):
        h, w = x.shape[-2:]
        # 获取 encoder 中间层特征
        e = self.encoder.get_intermediate_layers(
            x, n=self.intermediate_layer_idx['base']
        )

        # 如果 e 是 tuple，转换为 list
        if isinstance(e, tuple):
            e = list(e)

        # 遍历每个中间特征，进行 reshape + permute + FPN
        for i, fpn in enumerate(self.fpn_modules):
            # 动态获取 batch size 和通道数
            b, n, c = e[i].shape[:3] if e[i].ndim == 3 else (*e[i].shape[:2], e[i].shape[-1])
            # 计算 H 和 W 假设 n = H*W
            H = W = int((n)**0.5)
            e[i] = e[i].reshape(b, H, W, c).permute(0, 3, 1, 2).contiguous()
            e[i] = fpn(e[i])

        # Decoder
        ss = self.semsegdecoder.decode_head._forward_feature(e)
        out = self.semseghead(ss)
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









