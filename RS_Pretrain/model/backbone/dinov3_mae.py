# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.
"""
Minimal DinoViT implementation extracted from this repo to ease migration.

It keeps only the Vision Transformer backbone and the official pretrained
weight loader so you can drop the file into another project without pulling
the whole training stack.
"""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union
from urllib.parse import urlparse
from mmengine.logging import print_log
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
# from ..builder import BACKBONES
DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3"
logger = logging.getLogger("dinov3_minimal")


# -----------------------------------------------------------------------------
# Small utility helpers
# -----------------------------------------------------------------------------
def cat_keep_shapes(x_list: List[Tensor]) -> Tuple[Tensor, List[Tuple[int, ...]], List[int]]:
    shapes = [x.shape for x in x_list]
    num_tokens = [x.select(dim=-1, index=0).numel() for x in x_list]
    flattened = torch.cat([x.flatten(0, -2) for x in x_list])
    return flattened, shapes, num_tokens


def uncat_with_shapes(flattened: Tensor, shapes: List[Tuple[int, ...]], num_tokens: List[int]) -> List[Tensor]:
    outputs_splitted = torch.split_with_sizes(flattened, num_tokens, dim=0)
    shapes_adjusted = [shape[:-1] + torch.Size([flattened.shape[-1]]) for shape in shapes]
    outputs_reshaped = [o.reshape(shape) for o, shape in zip(outputs_splitted, shapes_adjusted)]
    return outputs_reshaped


def named_apply(
    fn: Callable,
    module: nn.Module,
    name: str = "",
    depth_first: bool = True,
    include_root: bool = False,
) -> nn.Module:
    if not depth_first and include_root:
        fn(module=module, name=name)
    for child_name, child_module in module.named_children():
        child_path = ".".join((name, child_name)) if name else child_name
        named_apply(
            fn=fn,
            module=child_module,
            name=child_path,
            depth_first=depth_first,
            include_root=True,
        )
    if depth_first and include_root:
        fn(module=module, name=name)
    return module


# -----------------------------------------------------------------------------
# Layers used by DinoViT
# -----------------------------------------------------------------------------
def make_2tuple(x):
    if isinstance(x, tuple):
        assert len(x) == 2
        return x

    assert isinstance(x, int)
    return (x, x)


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(torch.empty(dim, device=device))
        self.init_values = init_values

    def reset_parameters(self):
        nn.init.constant_(self.gamma, self.init_values)

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchEmbed(nn.Module):
    """
    2D image to patch embedding: (B,C,H,W) -> (B,N,D)
    """

    def __init__(
        self,
        img_size: Union[int, Tuple[int, int]] = 224,
        patch_size: Union[int, Tuple[int, int]] = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        norm_layer: Callable | None = None,
        flatten_embedding: bool = True,
    ) -> None:
        super().__init__()

        image_HW = make_2tuple(img_size)
        patch_HW = make_2tuple(patch_size)
        patch_grid_size = (
            image_HW[0] // patch_HW[0],
            image_HW[1] // patch_HW[1],
        )

        self.img_size = image_HW
        self.patch_size = patch_HW
        self.patches_resolution = patch_grid_size
        self.num_patches = patch_grid_size[0] * patch_grid_size[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.flatten_embedding = flatten_embedding

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_HW, stride=patch_HW)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.proj(x)  # B C H W
        H, W = x.size(2), x.size(3)
        x = x.flatten(2).transpose(1, 2)  # B HW C
        x = self.norm(x)
        if not self.flatten_embedding:
            x = x.reshape(-1, H, W, self.embed_dim)  # B H W C
        return x

    def flops(self) -> float:
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops

    def reset_parameters(self):
        k = 1 / (self.in_chans * (self.patch_size[0] ** 2))
        nn.init.uniform_(self.proj.weight, -math.sqrt(k), math.sqrt(k))
        if self.proj.bias is not None:
            nn.init.uniform_(self.proj.bias, -math.sqrt(k), math.sqrt(k))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def reset_parameters(self) -> None:
        nn.init.constant_(self.weight, 1)

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class Norm2d(nn.Module):
    """
    Channel-wise LayerNorm applied on (B, C, H, W).
    """

    def __init__(self, embed_dim: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=eps)

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)  # BCHW -> BHWC
        x = self.ln(x)
        return x.permute(0, 3, 1, 2).contiguous()


class RopePositionEmbedding(nn.Module):
    """
    RoPE positional embedding with no mixing of coordinates (axial) and no learnable weights.
    """

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int,
        base: float | None = 100.0,
        min_period: float | None = None,
        max_period: float | None = None,
        normalize_coords: Literal["min", "max", "separate"] = "separate",
        shift_coords: float | None = None,
        jitter_coords: float | None = None,
        rescale_coords: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ):
        super().__init__()
        assert embed_dim % (4 * num_heads) == 0
        both_periods = min_period is not None and max_period is not None
        if (base is None and not both_periods) or (base is not None and both_periods):
            raise ValueError("Either `base` or `min_period`+`max_period` must be provided.")

        D_head = embed_dim // num_heads
        self.base = base
        self.min_period = min_period
        self.max_period = max_period
        self.D_head = D_head
        self.normalize_coords = normalize_coords
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

        self.dtype = dtype  # Don't rely on self.periods.dtype
        self.register_buffer(
            "periods",
            torch.empty(D_head // 4, device=device, dtype=dtype),
            persistent=True,
        )
        self._init_weights()

    def forward(self, *, H: int, W: int) -> tuple[Tensor, Tensor]:
        device = self.periods.device
        dtype = self.dtype
        dd = {"device": device, "dtype": dtype}

        # Prepare coords in range [-1, +1]
        if self.normalize_coords == "max":
            max_HW = max(H, W)
            coords_h = torch.arange(0.5, H, **dd) / max_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / max_HW  # [W]
        elif self.normalize_coords == "min":
            min_HW = min(H, W)
            coords_h = torch.arange(0.5, H, **dd) / min_HW  # [H]
            coords_w = torch.arange(0.5, W, **dd) / min_HW  # [W]
        elif self.normalize_coords == "separate":
            coords_h = torch.arange(0.5, H, **dd) / H  # [H]
            coords_w = torch.arange(0.5, W, **dd) / W  # [W]
        else:
            raise ValueError(f"Unknown normalize_coords: {self.normalize_coords}")
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"), dim=-1)  # [H, W, 2]
        coords = coords.flatten(0, 1)  # [HW, 2]
        coords = 2.0 * coords - 1.0  # Shift range [0, 1] to [-1, +1]

        # Shift coords by adding a uniform value in [-shift, shift]
        if self.training and self.shift_coords is not None:
            shift_hw = torch.empty(2, **dd).uniform_(-self.shift_coords, self.shift_coords)
            coords += shift_hw[None, :]

        # Jitter coords by multiplying the range [-1, 1] by a log-uniform value in [1/jitter, jitter]
        if self.training and self.jitter_coords is not None:
            jitter_max = np.log(self.jitter_coords)
            jitter_min = -jitter_max
            jitter_hw = torch.empty(2, **dd).uniform_(jitter_min, jitter_max).exp()
            coords *= jitter_hw[None, :]

        # Rescale coords by multiplying the range [-1, 1] by a log-uniform value in [1/rescale, rescale]
        if self.training and self.rescale_coords is not None:
            rescale_max = np.log(self.rescale_coords)
            rescale_min = -rescale_max
            rescale_hw = torch.empty(1, **dd).uniform_(rescale_min, rescale_max).exp()
            coords *= rescale_hw

        # Prepare angles and sin/cos
        angles = 2 * math.pi * coords[:, :, None] / self.periods[None, None, :]  # [HW, 2, D//4]
        angles = angles.flatten(1, 2)  # [HW, D//2]
        angles = angles.tile(2)  # [HW, D]
        cos = torch.cos(angles)  # [HW, D]
        sin = torch.sin(angles)  # [HW, D]

        return (sin, cos)  # 2 * [HW, D]

    def _init_weights(self):
        device = self.periods.device
        dtype = self.dtype
        if self.base is not None:
            periods = self.base ** (
                2 * torch.arange(self.D_head // 4, device=device, dtype=dtype) / (self.D_head // 2)
            )  # [D//4]
        else:
            base = self.max_period / self.min_period
            exponents = torch.linspace(0, 1, self.D_head // 4, device=device, dtype=dtype)  # [D//4]
            periods = base**exponents  # range [1, max_period / min_period]
            periods = periods / base  # range [min_period / max_period, 1]
            periods = periods * self.max_period  # range [min_period, max_period]
        self.periods.data = periods


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class Mlp(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SwiGLUFFN(nn.Module, ListForwardMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias, device=device)

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)


def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        sdp = getattr(torch.nn.functional, "scaled_dot_product_attention", None)
        if sdp is not None:
            x = sdp(q, k, v)
        else:
            # Torch < 2.0 compatibility: manual attention
            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = torch.softmax(attn, dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError


# -----------------------------------------------------------------------------
# DinoViT backbone
# -----------------------------------------------------------------------------
ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()

# @BACKBONES.register_module()
class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        pretrained: str = 'dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth',
        arch: str = "vitb16",
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        enable_fpn: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.out_channels = [embed_dim, embed_dim, embed_dim, embed_dim]
        self.pretrained = pretrained
        self.arch = arch
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))
        self.enable_fpn = enable_fpn
        self.fpn_ops = self._make_fpn_ops(embed_dim) if enable_fpn else None
        # if isinstance(self.pretrained, str):
            # self._load_pretrained_weights()


    def init_weights(self, pretrained):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

        if isinstance(pretrained, str):
            state_dict = torch.load(pretrained, map_location="cpu")
            msg = self.load_state_dict(state_dict, strict=False)
            print(f"Loaded pretrained weights for {self.arch} with msg: {msg}")

    
    

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


    def prepare_tokens_with_masks(self, x: Tensor, masks=None, mask_ratio=0.0) -> Tuple[Tensor, Tuple[int, ...]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)
        ids_keep = None

        if mask_ratio!=0.0:
            x, ids_keep = self.random_masking(x, mask_ratio)
        
        if masks is not None:
            # x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            # x, ids_keep = self.random_masking(x, 0.0)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W), ids_keep

    def unpatchify_with_mask(self, feat: torch.Tensor, ids_keep: torch.Tensor) -> torch.Tensor:
        B, L_keep, C = feat.shape
        H, W = 32, 32
        L = H * W
        device = feat.device

        feat_full = self.mask_token.repeat(B, L, 1).to(device=device, dtype=feat.dtype)
        
        # 关键：扩展 ids_keep 到 [B, L_keep, C]
        ids_expanded = ids_keep.unsqueeze(-1).expand(-1, -1, C)  # [12, 1024, 768]
        
        # scatter 沿 dim=1（token 维度）
        feat_full.scatter_(1, ids_expanded, feat)
        
        return feat_full
        
    # def unpatchify_with_mask(self, feat, ids_keep):
    #     """
    #     Args:
    #         feat: [B, L_keep, C] 已保留的 token 特征
    #         ids_keep: [B, L_keep] 保留 token 的索引
    #         hw_shape: (H, W) 原始特征图的 patch 高宽

    #     Returns:
    #         feat_full: [B, C, H, W] 恢复到完整 feature map，包括 mask token
    #     """
    #     B, L_keep, C = feat.shape
    #     H, W = 32, 32
    #     L = H * W
    #     device = feat.device

    #     # 创建完整索引
    #     ids_full = torch.arange(L, device=device).unsqueeze(0).repeat(B, 1)  # [B, L]

    #     # 找出 mask token 的位置
    #     mask_positions = torch.ones_like(ids_full, dtype=torch.bool)
    #     mask_positions.scatter_(1, ids_keep, False)  # 保留位置设为 False

    #     # 创建 mask token
    #     num_mask = L - L_keep
        
    #     mask_tokens = self.mask_token.expand(B, num_mask, C).to(device)

    #     # 准备完整的 feat 容器
    #     feat_full = torch.zeros(B, L, C, device=device, dtype=feat.dtype)

    #     # 填充保留 token
    #     feat_full.scatter_(1, ids_keep.unsqueeze(-1).expand(-1, -1, C), feat)

    #     # 填充 mask token
    #     feat_full.scatter_(1, mask_positions.nonzero(as_tuple=False)[:,1].view(B, num_mask, 1).expand(-1, -1, C), mask_tokens)

    #     # reshape 回 feature map
    #     # feat_full = feat_full.transpose(1, 2).reshape(B, C, H, W)
    #     return feat_full

    # def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor], mask_ratio) -> List[Dict[str, Tensor]]:
    #     x = []
    #     rope = []
    #     ids_keeps = []
    #     for t_x, t_masks in zip(x_list, masks_list):
    #         t2_x, hw_tuple, ids_keep = self.prepare_tokens_with_masks(t_x, t_masks, mask_ratio)
    #         # print("ids_keep", ids_keep.shape)
    #         # print("x_masked", x_masked.shape)
    #         x.append(t2_x)
    #         rope.append(hw_tuple)
    #         ids_keeps.append(ids_keep)
    #     for _, blk in enumerate(self.blocks):
    #         if self.rope_embed is not None:
    #             # rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
    #             rope_sincos = []
    #             for (H, W), ids_keep in zip(rope, ids_keeps):
    #             # ✅ 正确：每次调用 rope_embed 返回 (sin, cos)
    #                 sin_full, cos_full = self.rope_embed(H=H, W=W)
    #                 if ids_keep is not None:
    #                     # 使用第一个样本的 ids_keep（因为 batch 共享 mask）
    #                     idx = ids_keep[0]  # shape: [L_keep]
    #                     sin = sin_full[idx]
    #                     cos = cos_full[idx]
    #                 else:
    #                     sin, cos = sin_full, cos_full
    #                 rope_sincos.append((sin, cos))
    #         else:
    #             rope_sincos = [None for _ in rope]
    #         # print("rope_sincos[0]: sin.shape =", rope_sincos[0][0].shape, "cos.shape =", rope_sincos[0][1].shape)
    #         x = blk(x, rope_sincos)
    #     all_x = x
    #     output = []
    #     for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
    #         if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
    #             if self.untie_global_and_local_cls_norm and self.training and idx == 1:
    #                 x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
    #             elif self.untie_cls_and_patch_norms:
    #                 x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
    #             else:
    #                 x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
    #             x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
    #         else:
    #             x_norm = self.norm(x)
    #             # print("x_norm", x_norm.shape)
    #             # x_norm = self.unpatchify_with_mask(x_norm, ids_keep)
    #             x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
    #             # print("x_norm_cls_reg", x_norm_cls_reg.shape)
    #             x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
    #             if mask_ratio!=0.0:
    #                 x_norm_patch = self.unpatchify_with_mask(x_norm_patch, ids_keep)
                
    #         output.append(
    #             {
    #                 "x_norm_clstoken": x_norm_cls_reg[:, 0],
    #                 "x_storage_tokens": x_norm_cls_reg[:, 1:],
    #                 "x_norm_patchtokens": x_norm_patch,
    #                 "x_prenorm": x,
    #                 "masks": masks,
    #                 "hw": rope[idx],
    #             }
    #         )
    #     return output

    def forward_features_list(self, x_list: List[torch.Tensor], masks_list: List[torch.Tensor], mask_ratio) -> List[Dict[str, torch.Tensor]]:
        # # --- 阶段1：掩码阶段（Masking & Token Preparation） ---
        # if torch.cuda.is_available():
        #     start_mask = torch.cuda.Event(enable_timing=True)
        #     end_mask = torch.cuda.Event(enable_timing=True)
        #     start_mask.record()

        x = []
        rope = []
        ids_keeps = []
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple, ids_keep = self.prepare_tokens_with_masks(t_x, t_masks, mask_ratio)
            x.append(t2_x)
            rope.append(hw_tuple)
            ids_keeps.append(ids_keep)

        # if torch.cuda.is_available():
        #     end_mask.record()
        #     torch.cuda.synchronize()  # 等待 GPU 完成
        #     time_mask = start_mask.elapsed_time(end_mask)  # 单位：毫秒
        # else:
        #     time_mask = 0.0

        # # --- 阶段2：编码阶段（Transformer Blocks） ---
        # if torch.cuda.is_available():
        #     start_enc = torch.cuda.Event(enable_timing=True)
        #     end_enc = torch.cuda.Event(enable_timing=True)
        #     start_enc.record()

        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = []
                for (H, W), ids_keep in zip(rope, ids_keeps):
                    sin_full, cos_full = self.rope_embed(H=H, W=W)
                    if ids_keep is not None:
                        idx = ids_keep[0]
                        sin = sin_full[idx]
                        cos = cos_full[idx]
                    else:
                        sin, cos = sin_full, cos_full
                    rope_sincos.append((sin, cos))
            else:
                rope_sincos = [None for _ in rope]
            x = blk(x, rope_sincos)

        # if torch.cuda.is_available():
        #     end_enc.record()
        #     torch.cuda.synchronize()
        #     time_enc = start_enc.elapsed_time(end_enc)
        # else:
        #     time_enc = 0.0

        all_x = x

        # # --- 阶段3：补全阶段（Unpatchify + Norm） ---
        # if torch.cuda.is_available():
        #     start_unpatch = torch.cuda.Event(enable_timing=True)
        #     end_unpatch = torch.cuda.Event(enable_timing=True)
        #     start_unpatch.record()

        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
                if mask_ratio != 0.0:
                    x_norm_patch = self.unpatchify_with_mask(x_norm_patch, ids_keep)  # 修正：传 masks + hw

            output.append({
                "x_norm_clstoken": x_norm_cls_reg[:, 0],
                "x_storage_tokens": x_norm_cls_reg[:, 1:],
                "x_norm_patchtokens": x_norm_patch,
                "x_prenorm": x,
                "masks": masks,
                "hw": rope[idx],
            })

        # if torch.cuda.is_available():
        #     end_unpatch.record()
        #     torch.cuda.synchronize()
        #     time_unpatch = start_unpatch.elapsed_time(end_unpatch)
        # else:
        #     time_unpatch = 0.0

        # --- 打印或记录时间（可选）---
        # print(f"[Timing] Masking: {time_mask:.2f} ms | Encoding: {time_enc:.2f} ms | Unpatchify: {time_unpatch:.2f} ms")

        # # 如果你想返回时间（用于日志/平均），可以存到类属性或额外返回
        # self.last_timing = {
        #     'masking_ms': time_mask,
        #     'encoding_ms': time_enc,
        #     'unpatchify_ms': time_unpatch
        # }

        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None, mask_ratio: Optional[torch.Tensor] = 0.0) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks], mask_ratio)[0]
        else:
            return self.forward_features_list(x, masks, mask_ratio)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    # def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
    #     ret = self.forward_features(*args, **kwargs)
    #     if is_training:
    #         return ret
    #     if isinstance(ret, list):
    #         ret = ret[0]
    #     return self.head(ret["x_norm_clstoken"])

    def _make_fpn_ops(self, embed_dim: int) -> Optional[Tuple[nn.Module, nn.Module, nn.Module, nn.Module]]:
        """
        Build FPN-like up/down-sampling heads mirroring the ViT code you provided.
        """
        if self.patch_size == 16:
            fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                Norm2d(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )
            fpn2 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            fpn3 = nn.Identity()
            fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif self.patch_size == 8:
            fpn1 = nn.Sequential(nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2))
            fpn2 = nn.Identity()
            fpn3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))
            fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=4, stride=4))
        else:
            logger.warning(f"No FPN preset for patch_size={self.patch_size}; using None.")
            return None
        return nn.ModuleList([fpn1, fpn2, fpn3, fpn4])

    def forward(
        self,
        x: torch.Tensor,
        masks: Optional[torch.Tensor] = None,
        mask_ratio: Optional[torch.Tensor] = 0.0,
    ) -> Tuple[torch.Tensor, ...]:
        """
        Exact deconv + LN + GELU / maxpool pathway like the ViT code you shared.

        Returns:
            Tuple of feature maps (B, C, H_i, W_i) from the four FPN branches.
        """
        if not self.enable_fpn or self.fpn_ops is None:
            raise ValueError("FPN is disabled. Recreate the model with enable_fpn=True to use this method.")

        feats = self.forward_features(x, mask_ratio=mask_ratio)
        if isinstance(feats, list):
            feats = feats[0]
        H, W = feats["hw"]
        patch_tokens = feats["x_norm_patchtokens"]  # (B, HW, C)
        B, _, C = patch_tokens.shape
        base_map = patch_tokens.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        outs = []
        for op in self.fpn_ops:
            outs.append(op(base_map))
        return tuple(outs)

# -----------------------------------------------------------------------------
# Pretrained weight helper
# -----------------------------------------------------------------------------
MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "vits16": {
        "model_kwargs": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=384,
            depth=12,
            num_heads=6,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
        "hashes": {"lvd1689m": "08c60483"},
        "weights_default": "lvd1689m",
    },
    "vits16plus": {
        "model_kwargs": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=384,
            depth=12,
            num_heads=6,
            ffn_ratio=6,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="swiglu",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
        "hashes": {"lvd1689m": "4057cbaa"},
        "weights_default": "lvd1689m",
    },
    "vitb16": {
        "model_kwargs": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=768,
            depth=12,
            num_heads=12,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
        "hashes": {"lvd1689m": "73cec8be"},
        "weights_default": "lvd1689m",
    },
    "vitl16": {
        "model_kwargs": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=1024,
            depth=24,
            num_heads=16,
            ffn_ratio=4,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="mlp",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
        "hashes": {"lvd1689m": "8aa4cbdd", "sat493m": "eadcf0ff"},
        "arch_overrides": {"sat493m": {"untie_global_and_local_cls_norm": True}},
        "weights_default": "lvd1689m",
    },
    "vitl16plus": {
        "model_kwargs": dict(
            img_size=224,
            patch_size=16,
            in_chans=3,
            pos_embed_rope_base=100,
            pos_embed_rope_normalize_coords="separate",
            pos_embed_rope_rescale_coords=2,
            pos_embed_rope_dtype="fp32",
            embed_dim=1024,
            depth=24,
            num_heads=16,
            ffn_ratio=6.0,
            qkv_bias=True,
            drop_path_rate=0.0,
            layerscale_init=1.0e-05,
            norm_layer="layernormbf16",
            ffn_layer="swiglu",
            ffn_bias=True,
            proj_bias=True,
            n_storage_tokens=4,
            mask_k_bias=True,
        ),
        "hashes": {"lvd1689m": "46503df0"},
        "weights_default": "lvd1689m",
    },
}


def _make_pretrained_url(arch: str, weights: str, hash_suffix: Optional[str]) -> str:
    hash_part = f"-{hash_suffix}" if hash_suffix else ""
    model_dir = f"dinov3_{arch}"
    model_filename = f"dinov3_{arch}_pretrain_{weights}{hash_part}.pth"
    return f"{DINOV3_BASE_URL}/{model_dir}/{model_filename}"


def build_dinov3_vit(
    arch: str = "vitb16",
    pretrained: bool = False,
    *,
    weights: Optional[str] = None,
    weights_path: Optional[str] = None,
    check_hash: bool = False,
    device: Optional[Union[torch.device, str]] = None,
    **override_kwargs,
) -> DinoVisionTransformer:
    """
    Build a DinoViT backbone and optionally load official pretrained weights.

    Args:
        arch: one of the keys in MODEL_CONFIGS, e.g. ``vitb16``.
        pretrained: if True, load weights from ``weights_path`` or the official URL.
        weights: weight flavor to use (defaults per arch, e.g. ``lvd1689m``).
        weights_path: local file path or URL to a state dict; overrides the default URL.
        check_hash: enable hash verification when downloading official weights.
        device: device used when constructing parameters.
        override_kwargs: forwarded to ``DinoVisionTransformer`` to tweak the config.
    """
    if arch not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported arch '{arch}'. Available: {list(MODEL_CONFIGS)}")

    cfg = MODEL_CONFIGS[arch]
    model_kwargs = dict(cfg["model_kwargs"])
    weight_key = weights or cfg.get("weights_default", "lvd1689m")
    if pretrained:
        arch_overrides = cfg.get("arch_overrides", {}).get(weight_key)
        if arch_overrides:
            model_kwargs.update(arch_overrides)
    model_kwargs.update(override_kwargs)

    model = DinoVisionTransformer(**model_kwargs, device=device)

    if pretrained:
        if weights_path is not None:
            parsed = urlparse(weights_path)
            if parsed.scheme in ("https", "http", "file"):
                state_dict = torch.hub.load_state_dict_from_url(
                    weights_path, map_location="cpu", check_hash=check_hash
                )
            else:
                state_dict = torch.load(weights_path, map_location="cpu")
        else:
            hash_suffix = cfg["hashes"].get(weight_key)
            url = _make_pretrained_url(arch=arch, weights=weight_key, hash_suffix=hash_suffix)
            state_dict = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=check_hash)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights for {arch} with msg: {msg}")
    else:
        model.init_weights()

    return model


__all__ = [
    "DinoVisionTransformer",
    "build_dinov3_vit",
    "MODEL_CONFIGS",
]


def DINOV3_ViT_B_MAE(args):
    backbone = DinoVisionTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="separate",
        pos_embed_rope_rescale_coords=2,
        pos_embed_rope_dtype="fp32",
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-05,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
        enable_fpn=True,
)
    return backbone
