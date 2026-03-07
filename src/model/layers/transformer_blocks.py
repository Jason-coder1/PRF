from typing import Optional
import torch
import torch.nn as nn
import numpy as np
np.float_ = np.float64
np.complex_ = np.complex128
from timm.models.layers import DropPath
from torch import Tensor
from .mln import MLN, nerf_positional_encoding



class Mlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.2,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.2,
        attn_drop=0.2,
        drop_path=0.2,
        act_layer=nn.GELU,
        #norm_layer=nn.LayerNorm,
        post_norm=False,
    ):
        super().__init__()
        self.post_norm = post_norm

        #self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        #self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        src2 = norm_layer(src)
        #src2 = self.norm1(src)
        src2 = self.attn(
            query=src2,
            key=src2,
            value=src2,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(norm_layer(src)))
        #src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward_post(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        src2 = self.attn(
            query=src,
            key=src,
            value=src,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(norm_layer(src2))
        src = src + self.drop_path2(norm_layer(self.mlp(src)))
        #src = src + self.drop_path1(self.norm1(src2))
        #src = src + self.drop_path2(self.norm2(self.mlp(src)))
        return src

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        if self.post_norm:
            return self.forward_post(
                src=src, mask=mask, key_padding_mask=key_padding_mask, norm_layer=norm_layer
            )

        return self.forward_pre(src=src, mask=mask, key_padding_mask=key_padding_mask, norm_layer=norm_layer)


class Cross_Block(nn.Module):
    def __init__(
        self,
        dim=128,
        num_heads=8,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.2,
        attn_drop=0.2,
        drop_path=0.2,
        act_layer=nn.GELU,
        #norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        #self.norm1 = norm_layer(dim)
        self.attn = torch.nn.MultiheadAttention(
            dim,
            num_heads=num_heads,
            add_bias_kv=qkv_bias,
            dropout=attn_drop,
            batch_first=True,
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        #self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward_pre(
        self,
        src,
        src_kv,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        norm_layer: Optional[nn.Module] = None,
    ):
        src2 = norm_layer(src)
        src2_kv = norm_layer(src_kv)
        #src2 = self.norm1(src)
        #src2_kv = self.norm1(src_kv)
        src2 = self.attn(
            query=src2,
            key=src2_kv,
            value=src2_kv,
            attn_mask=mask,
            key_padding_mask=key_padding_mask,
        )[0]
        src = src + self.drop_path1(src2)
        src = src + self.drop_path2(self.mlp(norm_layer(src)))
        #src = src + self.drop_path2(self.mlp(self.norm2(src)))
        return src

    def forward(
        self,
        src,
        src_kv,
        mask: Optional[Tensor] = None,
        key_padding_mask: Optional[Tensor] = None,
        norm_layer: Optional[nn.Module] = None,
    ):

        return self.forward_pre(src=src, src_kv=src_kv, mask=mask, key_padding_mask=key_padding_mask, norm_layer=norm_layer)
