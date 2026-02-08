# Copyright (c) OpenMMLab. All rights reserved.
import math
import warnings

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmengine.logging import print_log
from mmengine.model import BaseModule, ModuleList
from mmengine.model.weight_init import (constant_init, kaiming_init,
                                        trunc_normal_)
from mmengine.runner.checkpoint import CheckpointLoader, load_state_dict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.utils import _pair as to_2tuple
import torch.nn.functional as F
from mmseg.registry import MODELS
from mmseg.models.utils import PatchEmbed, resize
from mmengine.dist import get_dist_info


class PatchEmbed(PatchEmbed):
    def __init__(self, img_size, in_channels=3, embed_dims=768, conv_type='Conv2d', kernel_size=16, stride=None, padding='corner', dilation=1, bias=True, norm_cfg=None, input_size=None, init_cfg=None,):
        super().__init__(in_channels, embed_dims, conv_type, kernel_size, stride, padding, dilation, bias, norm_cfg, input_size, init_cfg)
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        self.patch_shape = (img_size[0] // kernel_size[0], img_size[1] // kernel_size[1])

        self.num_patches = (img_size[1] // kernel_size[1]) * (img_size[0] // kernel_size[0])

    def forward(self, x):

        if self.adap_padding:
            x = self.adap_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x, out_size
        

class TransformerEncoderLayer(BaseModule):
    """Implements one encoder layer in Vision Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        drop_rate (float): Probability of an element to be zeroed
            after the feed forward layer. Default: 0.0.
        attn_drop_rate (float): The drop out rate for attention layer.
            Default: 0.0.
        drop_path_rate (float): stochastic depth rate. Default 0.0.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        qkv_bias (bool): enable bias for qkv if True. Default: True
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default: True.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 feedforward_channels,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 num_fcs=2,
                 qkv_bias=True,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 batch_first=True,
                 attn_cfg=dict(),
                 ffn_cfg=dict(),
                 with_cp=False):
        super().__init__()

        self.norm1_name, norm1 = build_norm_layer(
            norm_cfg, embed_dims, postfix=1)
        self.add_module(self.norm1_name, norm1)

        attn_cfg.update(
            dict(
                embed_dims=embed_dims,
                num_heads=num_heads,
                attn_drop=attn_drop_rate,
                proj_drop=drop_rate,
                batch_first=batch_first,
                bias=qkv_bias))

        self.build_attn(attn_cfg)

        self.norm2_name, norm2 = build_norm_layer(
            norm_cfg, embed_dims, postfix=2)
        self.add_module(self.norm2_name, norm2)

        ffn_cfg.update(
            dict(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate)
                if drop_path_rate > 0 else None,
                act_cfg=act_cfg))
        self.build_ffn(ffn_cfg)
        self.with_cp = with_cp

    def build_attn(self, attn_cfg):
        self.attn = MultiheadAttention(**attn_cfg)

    def build_ffn(self, ffn_cfg):
        self.ffn = FFN(**ffn_cfg)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):

        def _inner_forward(x):
            x = self.attn(self.norm1(x), identity=x)
            x = self.ffn(self.norm2(x), identity=x)
            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)
        return x

class Norm2d(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=1e-6)
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x
    
# @MODELS.register_module()
class ViT(BaseModule):
    """Vision Transformer.

    This backbone is the implementation of `An Image is Worth 16x16 Words:
    Transformers for Image Recognition at
    Scale <https://arxiv.org/abs/2010.11929>`_.

    Args:
        img_size (int | tuple): Input image size. Default: 224.
        patch_size (int): The patch size. Default: 16.
        in_channels (int): Number of input channels. Default: 3.
        embed_dims (int): embedding dimension. Default: 768.
        num_layers (int): depth of transformer. Default: 12.
        num_heads (int): number of attention heads. Default: 12.
        mlp_ratio (int): ratio of mlp hidden dim to embedding dim.
            Default: 4.
        out_indices (list | tuple | int): Output from which stages.
            Default: -1.
        qkv_bias (bool): enable bias for qkv if True. Default: True.
        drop_rate (float): Probability of an element to be zeroed.
            Default 0.0
        attn_drop_rate (float): The drop out rate for attention layer.
            Default 0.0
        drop_path_rate (float): stochastic depth rate. Default 0.0
        with_cls_token (bool): Whether concatenating class token into image
            tokens as transformer input. Default: True.
        output_cls_token (bool): Whether output the cls_token. If set True,
            `with_cls_token` must be True. Default: False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN')
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='GELU').
        patch_norm (bool): Whether to add a norm in PatchEmbed Block.
            Default: False.
        final_norm (bool): Whether to add a additional layer to normalize
            final feature map. Default: False.
        interpolate_mode (str): Select the interpolate mode for position
            embeding vector resize. Default: bicubic.
        num_fcs (int): The number of fully-connected layers for FFNs.
            Default: 2.
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save
            some memory while slowing down the training speed. Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None.
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dims=768,
                 num_layers=12,
                 num_heads=12,
                 mlp_ratio=4,
                 out_indices=-1,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 with_cls_token=False,
                 output_cls_token=False,
                 norm_cfg=dict(type='LN'),
                 act_cfg=dict(type='GELU'),
                 patch_norm=False,
                 final_norm=False,
                 interpolate_mode='bicubic',
                 num_fcs=2,
                 norm_eval=False,
                 with_cp=False,
                 pretrained=None,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.in_channels = in_channels
        self.out_channels = [embed_dims, embed_dims, embed_dims, embed_dims]
        if isinstance(img_size, int):
            img_size = to_2tuple(img_size)
        elif isinstance(img_size, tuple):
            if len(img_size) == 1:
                img_size = to_2tuple(img_size[0])
            assert len(img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(img_size)}'

        if output_cls_token:
            assert with_cls_token is True, f'with_cls_token must be True if' \
                f'set output_cls_token to True, but got {with_cls_token}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')

        self.img_size = img_size
        self.patch_size = patch_size
        self.interpolate_mode = interpolate_mode
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        self.pretrained = pretrained

        self.patch_embed = PatchEmbed(
            img_size = img_size,
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,
            padding='corner',
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None,
        )

        num_patches = (img_size[0] // patch_size) * \
            (img_size[1] // patch_size)

        self.with_cls_token = with_cls_token
        self.output_cls_token = output_cls_token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dims))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dims))
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            if out_indices == -1:
                out_indices = num_layers - 1
            self.out_indices = [out_indices]
        elif isinstance(out_indices, list) or isinstance(out_indices, tuple):
            self.out_indices = out_indices
        else:
            raise TypeError('out_indices must be type of int, list or tuple')

        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, num_layers)
        ]  # stochastic depth decay rule

        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(
                TransformerEncoderLayer(
                    embed_dims=embed_dims,
                    num_heads=num_heads,
                    feedforward_channels=mlp_ratio * embed_dims,
                    attn_drop_rate=attn_drop_rate,
                    drop_rate=drop_rate,
                    drop_path_rate=dpr[i],
                    num_fcs=num_fcs,
                    qkv_bias=qkv_bias,
                    act_cfg=act_cfg,
                    norm_cfg=norm_cfg,
                    with_cp=with_cp,
                    batch_first=True))

        self.final_norm = final_norm
        if final_norm:
            self.norm1_name, norm1 = build_norm_layer(
                norm_cfg, embed_dims, postfix=1)
            self.add_module(self.norm1_name, norm1)

        if patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
                Norm2d(embed_dims),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dims, embed_dims, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)


################################ MAE Initialization ################################# 
    def init_weights(self, pretrained):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        pretrained = pretrained or self.pretrained
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)

            checkpoint = torch.load(pretrained, map_location='cpu')

            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint

            if 'mask_token' in state_dict:
                del state_dict['mask_token']

            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            state_dict = {k.replace("blocks","layers"): v for k, v in state_dict.items()}
            state_dict = {k.replace("norm","ln"): v for k, v in state_dict.items()}

            # for MoBY, load model of online branch
            if sorted(list(state_dict.keys()))[0].startswith('encoder'):
                state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

            # remove patch embed when inchan != 3

            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key
                # 调整attn模块中的键名
                if "patch_embed.proj" in key:
                    new_key = key.replace("patch_embed.proj", "patch_embed.projection")


                if ".attn.qkv." in key:
                    new_key = key.replace(".attn.qkv.", ".attn.attn.in_proj_")
                if ".attn.proj." in key:
                    new_key = key.replace(".attn.proj.", ".attn.attn.out_proj.")
                
                # 调整ffn模块中的键名
                if ".mlp.fc1." in key:
                    new_key = key.replace(".mlp.fc1.", ".ffn.layers.0.0.")
                if ".mlp.fc2." in key:
                    new_key = key.replace(".mlp.fc2.", ".ffn.layers.1.")
                
                new_state_dict[new_key] = value

            state_dict = new_state_dict


            # for name in state_dict.keys():
            #     print(f"dict_Name: {name}")

            if self.in_channels != 3:
                for k in list(state_dict.keys()):
                    if 'patch_embed.proj' in k:
                        del state_dict[k]

            # print('$$$$$$$$$$$$$$$$$')
            # print(state_dict.keys())

            # print('#################')
            # print(self.state_dict().keys())

            rank, _ = get_dist_info()
            if 'pos_embed' in state_dict:
                pos_embed_checkpoint = state_dict['pos_embed']
                embedding_size = pos_embed_checkpoint.shape[-1]
                H, W = self.patch_embed.patch_shape
                num_patches = self.patch_embed.num_patches
                num_extra_tokens = 1
                # height (== width) for the checkpoint position embedding
                orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
                # height (== width) for the new position embedding
                new_size = int(num_patches ** 0.5)
                # class_token and dist_token are kept unchanged
                if orig_size != new_size:
                    if rank == 0:
                        print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, H, W))
                    # extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
                    # only the position tokens are interpolated
                    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
                    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).contiguous()
                    pos_tokens = torch.nn.functional.interpolate(
                        pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
                    new_pos_embed = pos_tokens.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
                    # new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
                    state_dict['pos_embed'] = new_pos_embed
                else:
                    state_dict['pos_embed'] = pos_embed_checkpoint[:, num_extra_tokens:]

            msg = self.load_state_dict(state_dict, False)

            if rank == 0:
                print(msg[0])

        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')


    # def init_weights(self, pretrained):
    #     """Initialize the weights in backbone.

    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """
    #     pretrained = pretrained
    #     rank, _ = get_dist_info()

    #     def _init_weights(m):
    #         if isinstance(m, nn.Linear):
    #             trunc_normal_(m.weight, std=.02)
    #             if isinstance(m, nn.Linear) and m.bias is not None:
    #                 nn.init.constant_(m.bias, 0)
    #         elif isinstance(m, nn.LayerNorm):
    #             nn.init.constant_(m.bias, 0)
    #             nn.init.constant_(m.weight, 1.0)

    #     if isinstance(pretrained, str):
    #         self.apply(_init_weights)

    #         checkpoint = torch.load(pretrained, map_location='cpu')

    #         if 'state_dict' in checkpoint:
    #             state_dict = checkpoint['state_dict']
    #         elif 'model' in checkpoint:
    #             state_dict = checkpoint['model']
    #         else:
    #             state_dict = checkpoint

    #         if list(state_dict.keys())[0].startswith('module.'):
    #             state_dict = {k[7:]: v for k, v in state_dict.items()}
 
    #         if 'pos_embed' in state_dict:
    #             pos_embed_checkpoint = state_dict['pos_embed']
    #             embedding_size = pos_embed_checkpoint.shape[-1]
    #             H, W = self.patch_embed.patch_shape
    #             num_patches = self.patch_embed.num_patches
    #             num_extra_tokens = 0
    #             orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    #             new_size = int(num_patches ** 0.5)
    #             if orig_size != new_size:
    #                 if rank == 0:
    #                     print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, H, W))
    #                 pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    #                 pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2).contiguous()
    #                 pos_tokens = torch.nn.functional.interpolate(
    #                     pos_tokens, size=(H, W), mode='bicubic', align_corners=False)
    #                 new_pos_embed = pos_tokens.permute(0, 2, 3, 1).contiguous().flatten(1, 2)
    #                 state_dict['pos_embed'] = new_pos_embed
    #             else:
    #                 state_dict['pos_embed'] = pos_embed_checkpoint[:, num_extra_tokens:]

    #         msg = self.load_state_dict(state_dict, False)
            
    #         if rank == 0:
    #             print(msg[0])

    def entropy_guided_masking(self, x, pred, mask_ratio, patch_size, eps=1e-8):
        """
        Args:
            x: [B, L, D] token 特征
            pred: [B, C, H, W] 模型预测概率（经过 softmax）
            mask_ratio: 掩码比例
            patch_size: ViT patch 尺度
        Returns:
            x_keep: 保留的 token 特征
            ids_keep: 保留 token 的索引
        """

        B, L, D = x.shape
        device = x.device

        # ---- Step 1: 计算像素级熵 ----
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=eps, max=1.0)
        entropy_map = -torch.sum(pred * torch.log(pred + eps), dim=1, keepdim=True)  # [B, 1, H, W]
        # print("pred", pred.shape)
        # print("entropy_map", entropy_map)
        # ---- Step 2: 按 patch 统计熵值 ----
        H, W = pred.shape[2], pred.shape[3]
        h_p, w_p = H // patch_size, W // patch_size
        entropy_map = F.interpolate(entropy_map, size=(h_p, w_p), mode='bilinear', align_corners=False)
        patch_entropy = entropy_map.flatten(1)  # [B, L]
        
        # ---- Step 3: 基于熵排序并选择 token ----
        # len_keep = int(L * (1 - mask_ratio))
        len_keep = int(L * (1 - mask_ratio))
        ids_sorted = torch.argsort(patch_entropy, dim=1, descending=True)  # 高熵优先
        # print("ids_sorted", ids_sorted[0, :10])
        ids_keep = ids_sorted[:, :len_keep]  # 保留高熵 token
        # print("ids_keep", ids_keep)
        # ---- Step 4: 按原顺序排序，保持空间一致性 ----
        ids_keep, _ = torch.sort(ids_keep, dim=1)
        # print("ids_keep", ids_keep)
        # ---- Step 5: 选择保留的 token ----
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_keep, ids_keep


    def uniform_masking(self, x, mask_ratio):
        """
        在 token 序列中均匀采样，确保任意掩码率都均匀覆盖整幅图像。
        Args:
            x: [B, L, D] token 特征
            mask_ratio: 掩码比例
        Returns:
            x_keep: 保留的 token 特征
            ids_keep: 保留 token 的索引
        """
        B, L, D = x.shape
        device = x.device
        len_keep = int(L * (1 - mask_ratio))

        # ---- Step 1: 均匀采样索引 ----
        if len_keep <= 0:
            raise ValueError(f"mask_ratio={mask_ratio} 导致 len_keep=0")

        step = L / len_keep
        ids_keep_single = torch.floor(torch.arange(len_keep, device=device) * step).long()
        ids_keep_single = torch.clamp(ids_keep_single, max=L - 1)  # 防止越界

        # ---- Step 2: 扩展到 batch ----
        ids_keep = ids_keep_single.unsqueeze(0).repeat(B, 1)  # [B, len_keep]

        # ---- Step 3: 提取 token ----
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_keep, ids_keep

    def random_masking(self, x, mask_ratio):

        B, L, D = x.shape
        device = x.device
        len_keep = int(L * (1 - mask_ratio))

        perm = torch.randperm(L, device=device)
        ids_keep_single = perm[:len_keep]  # [len_keep]
        ids_keep = ids_keep_single.unsqueeze(0).repeat(B, 1)  # [B, len_keep]

        # 按原顺序排序，保持token顺序不变
        ids_keep, _ = torch.sort(ids_keep, dim=1)

        # 挑选保留token
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_keep, ids_keep


    def unpatchify_with_mask(self, feat, ids_keep, hw_shape):
        """
        Args:
            feat: [B, L_keep, C] 已保留的 token 特征
            ids_keep: [B, L_keep] 保留 token 的索引
            hw_shape: (H, W) 原始特征图的 patch 高宽

        Returns:
            feat_full: [B, C, H, W] 恢复到完整 feature map，包括 mask token
        """
        B, L_keep, C = feat.shape
        H, W = hw_shape
        L = H * W
        device = feat.device

        # 创建完整索引
        ids_full = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)  # [B, L]

        # 找出 mask token 的位置
        mask_positions = torch.ones_like(ids_full, dtype=torch.bool)
        mask_positions.scatter_(1, ids_keep, False)  # 保留位置设为 False

        # 创建 mask token
        num_mask = L - L_keep
        mask_tokens = self.mask_token.expand(B, num_mask, C).to(device)

        # 准备完整的 feat 容器
        feat_full = torch.zeros(B, L, C, device=device, dtype=feat.dtype)

        # 填充保留 token
        feat_full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, C), feat)

        # 填充 mask token
        feat_full.scatter_(1, mask_positions.nonzero(as_tuple=False)[:,1].view(B, num_mask, 1).expand(-1, -1, C), mask_tokens)

        # reshape 回 feature map
        # feat_full = feat_full.transpose(1, 2).reshape(B, C, H, W)
        return feat_full




    def class_entropy_guided_masking(self, x, pred, mask_ratio, patch_size, high_entropy_ratio=0.5, eps=1e-8):
        """
        Class-aware entropy guided masking (batch-consistent)

        Args:
            x: [B, L, D]                 # token embeddings
            pred: [B, C, H, W]           # model prediction logits (before or after softmax)
            mask_ratio: float            # ratio of tokens to mask
            patch_size: int              # patch size (e.g. 16)
            high_entropy_ratio: float    # proportion of high-entropy tokens kept per class
            eps: float                   # small value to avoid log(0)
        Returns:
            x_keep: [B, L_keep, D]       # retained token embeddings
            ids_keep: [B, L_keep]        # indices of kept tokens
        """

        B, L, D = x.shape
        device = x.device

        # ---- Step 1: 计算像素级 softmax 与熵 ----
        pred_prob = F.softmax(pred, dim=1).clamp(min=eps, max=1.0)
        entropy_map = -torch.sum(pred_prob * torch.log(pred_prob + eps), dim=1, keepdim=True)  # [B, 1, H, W]

        # ---- Step 2: 生成伪标签 (argmax over pred) ----
        pseudo_label = pred_prob.argmax(dim=1, keepdim=True).float()  # [B, 1, H, W]

        # ---- Step 3: 降采样到 patch 尺度 ----
        H, W = pred.shape[2], pred.shape[3]
        h_p, w_p = H // patch_size, W // patch_size
        patch_entropy = F.interpolate(entropy_map, size=(h_p, w_p), mode='bilinear', align_corners=False).flatten(1)  # [B, L]
        patch_label = F.interpolate(pseudo_label, size=(h_p, w_p), mode='nearest').flatten(1).long()  # [B, L]

        # ---- Step 4: 全 batch 统一保留 token 数量 ----
        len_keep = int(L * (1 - mask_ratio))
        ids_keep_all = []

        for b in range(B):
            entropy_b = patch_entropy[b]  # [L]
            label_b = patch_label[b]      # [L]
            all_indices = torch.arange(L, device=device)

            selected_indices = []

            # ---- 遍历每个类别（前景/背景等） ----
            for cls in label_b.unique():
                cls = cls.item()
                cls_mask = (label_b == cls)
                cls_indices = all_indices[cls_mask]
                if cls_indices.numel() == 0:
                    continue

                cls_entropy = entropy_b[cls_mask]
                num_cls_keep = int(len_keep / label_b.unique().numel())  # 每类平分保留数量
                num_high_entropy = int(num_cls_keep * high_entropy_ratio)

                # 高熵优先选
                _, sorted_idx = torch.sort(cls_entropy, descending=True)
                high_entropy_keep = cls_indices[sorted_idx[:num_high_entropy]]

                # 剩余随机选
                remain_indices = cls_indices[sorted_idx[num_high_entropy:]]
                if remain_indices.numel() > 0:
                    rand_perm = torch.randperm(remain_indices.numel(), device=device)
                    rand_keep = remain_indices[rand_perm[:num_cls_keep - num_high_entropy]]
                    cls_keep = torch.cat([high_entropy_keep, rand_keep], dim=0)
                else:
                    cls_keep = high_entropy_keep

                selected_indices.append(cls_keep)

            # ---- 拼接并补齐数量（不足时随机补齐） ----
            ids_keep_b = torch.cat(selected_indices, dim=0)
            if ids_keep_b.numel() < len_keep:
                remaining = torch.tensor(list(set(all_indices.tolist()) - set(ids_keep_b.tolist())), device=device)
                if remaining.numel() > 0:
                    rand_extra = remaining[torch.randperm(remaining.numel(), device=device)[:len_keep - ids_keep_b.numel()]]
                    ids_keep_b = torch.cat([ids_keep_b, rand_extra], dim=0)
            elif ids_keep_b.numel() > len_keep:
                ids_keep_b = ids_keep_b[:len_keep]

            ids_keep_b, _ = torch.sort(ids_keep_b)
            ids_keep_all.append(ids_keep_b)

        # ---- Step 5: 拼 batch ----
        ids_keep = torch.stack(ids_keep_all, dim=0)  # [B, len_keep]
        x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))

        return x_keep, ids_keep

    def _pos_embeding(self, patched_img, hw_shape, pos_embed):
        """Positioning embeding method.

        Resize the pos_embed, if the input image size doesn't match
            the training size.
        Args:
            patched_img (torch.Tensor): The patched image, it should be
                shape of [B, L1, C].
            hw_shape (tuple): The downsampled image resolution.
            pos_embed (torch.Tensor): The pos_embed weighs, it should be
                shape of [B, L2, c].
        Return:
            torch.Tensor: The pos encoded image feature.
        """
        assert patched_img.ndim == 3 and pos_embed.ndim == 3, \
            'the shapes of patched_img and pos_embed must be [B, L, C]'
        x_len, pos_len = patched_img.shape[1], pos_embed.shape[1]
        if x_len != pos_len:
            if pos_len == (self.img_size[0] // self.patch_size) * (
                    self.img_size[1] // self.patch_size) + 1:
                pos_h = self.img_size[0] // self.patch_size
                pos_w = self.img_size[1] // self.patch_size
            else:
                raise ValueError(
                    'Unexpected shape of pos_embed, got {}.'.format(
                        pos_embed.shape))
            pos_embed = self.resize_pos_embed(pos_embed, hw_shape,
                                              (pos_h, pos_w),
                                              self.interpolate_mode)
        return self.drop_after_pos(patched_img + pos_embed)

    @staticmethod
    def resize_pos_embed(pos_embed, input_shpae, pos_shape, mode):
        """Resize pos_embed weights.

        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, L_new, C]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h, pos_w = pos_shape
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2).contiguous()
        pos_embed_weight = resize(
            pos_embed_weight, size=input_shpae, align_corners=False, mode=mode)
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2).contiguous()
        pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed

    def forward(self, inputs, mask_ratio=0):
        # inputs shape: B, 3, H, W，这里以B，3，512，512为例
        B = inputs.shape[0]
        x, hw_shape = self.patch_embed(inputs) # x shape: B, L, C， L=(H//p)*(W//P)，C=#*P*P，x shape: B, 1024, 768
        # print("x shape", x.shape)
        if mask_ratio !=0:
            # x_masked, ids_keep = self.uniform_masking(x, mask_ratio=mask_ratio)
            x_masked, ids_keep = self.random_masking(x, mask_ratio=mask_ratio)
            # x_masked, ids_keep = self. entropy_guided_masking(x, pred, mask_ratio, self.patch_size)
            # x_masked, ids_keep = self.class_entropy_guided_masking(x, pred, mask_ratio, self.patch_size)

        else:
            x_masked = x
        # pixel_mask = self.build_pixel_mask_from_ids(ids_keep=ids_keep, device=x.device)


        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_masked = torch.cat((cls_tokens, x_masked), dim=1)

        if not self.with_cls_token:
            x_masked = x_masked[:, 1:]

        if mask_ratio !=0:
            x_masked = x_masked + self.pos_embed[:, ids_keep[0], :]
        else:
            x_masked = x_masked + self.pos_embed
            ids_keep = None

        x_masked = self.drop_after_pos(x_masked)

        features_kd = []
        outs = []
        for i, layer in enumerate(self.layers):
            x_masked = layer(x_masked)
            if i == len(self.layers) - 1:
                if self.final_norm:
                    x_masked = self.norm1(x_masked)
            if i in self.out_indices:
                if self.with_cls_token:
                    # Remove class token and reshape token for decoder head
                    out = x_masked[:, 1:]
                else:
                    out = x_masked
                B, _, C = out.shape
                features_kd.append(out)
                if mask_ratio !=0:
                    out = self.unpatchify_with_mask(out, ids_keep, hw_shape)
                # features_kd.append(out)
                out = out.reshape(B, hw_shape[0], hw_shape[1], C).permute(0, 3, 1, 2).contiguous()
                if self.output_cls_token:
                    out = [out, x_masked[:, 0]]
                # print("out shape", out.shape) out shape: B, C, P, P， out shape: B, 768, 32, 32，
                outs.append(out)
        ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4] #这里的FPN为反卷积和池化模块
        for i in range(len(ops)):
            outs[i] = ops[i](outs[i])
            # 得到的几个特征图的大小分别为：
            # B, 768, 128, 128
            # B, 768, 64, 64
            # B, 768, 32, 32
            # B, 768, 16, 16
            #以上几个不同尺度的特征图送人UperNet进行解码
        return tuple(outs)



    def train(self, mode=True):
        super().train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.LayerNorm):
                    m.eval()

def ViT_MAE_B(args):
    backbone = ViT(
        img_size=args.image_size,
        in_channels=3,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=[3, 5, 7, 11],
        embed_dims=768,
        num_layers=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0., 
        # with_cp=True,      
        with_cp=False,      
    )
    return backbone

def ViT_MAE_L(args):
    backbone = ViT(
        img_size=args.image_size,
        in_channels=3,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=[7, 11, 15, 23],
        embed_dims=1024,
        num_layers=24,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        with_cp=True,    
    )
    return backbone


def ViT_H(args):
    backbone = ViT(
        img_size=args.image_size,
        in_channels=3,
        patch_size=16,
        drop_path_rate=0.1,
        out_indices=[15, 23, 27, 31],
        embed_dims=1280,
        num_layers=32,
        num_heads=16,
        mlp_ratio=4,
        qkv_bias=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        with_cp=True,    
    )
    return backbone




if __name__ == "__main__":
    model = ViT(
            img_size=256,
            in_channels=3,
            patch_size=16,
            drop_path_rate=0.1,
            out_indices=[3, 5, 7, 11],
            embed_dims=768,
            num_layers=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            drop_rate=0.,
            attn_drop_rate=0.,    
        )

    
    input = torch.randn(4,3,256,256)
    output = model(input)

    for i in range(len(output)):
        print(i, output[i].shape)