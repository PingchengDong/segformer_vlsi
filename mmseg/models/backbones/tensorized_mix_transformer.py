# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from mmseg.t3nsor.layers import TTLinear
from mmseg.tensorized_rnn.lstm import LSTMCell, LSTM
from mmseg.tensorized_rnn.tt_linearset import TTLinearSet
from mmseg.tensorized_rnn.rnn_utils import tt_shape
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math

def window_partition(x, window_size: List[int], img_size: List[int]):
    B, N, C = x.shape
    H = img_size[0]
    W = img_size[1]
    assert(H % window_size[0] == 0, f'height ({H}) must be divisible by window ({window_size[0]})')
    assert(W % window_size[1] == 0, '')
    # B, H, W, C -> B, H/win, win, W/win, win, C
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    # B, H/win, win, W/win, win, C -> B, H/win, W/win, win, win, C
    # B, H/win, W/win, win, win, C -> B*(H/win)*(W/win), win*win, C
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0]*window_size[1], C)
    return windows


def window_reverse(windows, window_size: List[int], img_size: List[int]):
    H, W = img_size
    B = int(windows.shape[0] / (H * W / window_size[0] / window_size[1]))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, -1)
    return x


def grid_partition(x, grid_size: List[int],  img_size: List[int]):
    B, N, C = x.shape
    H = img_size[0]
    W = img_size[1]
    assert(H % grid_size[0] == 0, f'height {H} must be divisible by grid {grid_size[0]}')
    assert(W % grid_size[1] == 0, '')
    # B, H, W, C -> B, grid, H/grid, grid, W/grid, C
    x = x.view(B, grid_size[0], H // grid_size[0], grid_size[1], W // grid_size[1], C)
    # B, grid, H/grid, grid, W/grid, C -> B, H/grid, W/grid, grid, grid, C
    # B, grid, H/grid, grid, W/grid, C -> B*(H/grid)*(W/grid), grid*grid, C
    windows = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, grid_size[0]*grid_size[1], C)
    return windows


def grid_reverse(windows, grid_size: List[int], img_size: List[int]):
    H, W = img_size
    B = int(windows.shape[0] / (H * W / grid_size[0] / grid_size[1]))
    # view: [B*num_windows, Mh, Mw, C] -> [B, H//Mh, W//Mw, Mh, Mw, C]
    x = windows.view(B, H // grid_size[0], W // grid_size[1], grid_size[0], grid_size[1], -1)
    # permute: [B, H//Mh, W//Mw, Mh, Mw, C] -> [B, H//Mh, Mh, W//Mw, Mw, C]
    # view: [B, H//Mh, Mh, W//Mw, Mw, C] -> [B, H, W, C]
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H * W, -1)
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 is_naive=False,  is_tt=True, n_cores=4, tt_rank=8):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.is_naive = is_naive
        self.is_tt = is_tt
        self.n_gate = 1
        self.n_cores = n_cores
        self.tt_rank = tt_rank

        if self.is_naive:
            self.TTfc1 = TTLinearSet(in_features=in_features,
                                     out_features=hidden_features, n_gates=self.n_gate,
                                     bias=True, auto_shapes=True,
                                     d=self.n_cores, tt_rank=self.tt_rank)

            self.TTfc2 = TTLinearSet(in_features=hidden_features,
                                     out_features=out_features, n_gates=self.n_gate,
                                     bias=True, auto_shapes=True,
                                     d=self.n_cores, tt_rank=self.tt_rank)
        else:
            shape_fc1 = tt_shape(in_features, hidden_features,
                             self.n_cores, self.n_gate, new_core=None)
            self.TTfc1 = TTLinear(out_features=self.n_gate * hidden_features, shape=shape_fc1,
                                  bias=True, auto_shapes=False, d=self.n_cores,
                                  tt_rank=self.tt_rank)

            shape_fc2 = tt_shape(hidden_features, out_features,
                             self.n_cores, self.n_gate, new_core=None)
            self.TTfc2 = TTLinear(out_features=self.n_gate * out_features, shape=shape_fc2,
                                  bias=True, auto_shapes=False, d=self.n_cores,
                                  tt_rank=self.tt_rank)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        if self.is_tt:
            x = x.reshape(B * N, C)
            x = self.TTfc1(x).reshape(B, N, -1)
            x = self.dwconv(x, H, W)
            x = self.act(x)
            x = self.drop(x)
            x = x.reshape(B * N, -1)
            x = self.TTfc2(x).reshape(B, N, -1)
            x = self.drop(x)
        else:
            x = self.fc1(x)
            x = self.dwconv(x, H, W)
            x = self.act(x)
            x = self.drop(x)
            x = self.fc2(x)
            x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 is_naive=False,  is_tt=True, n_cores=4, tt_rank=8):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        #tensor train
        self.is_naive=is_naive
        self.is_tt=is_tt
        self.n_gate = 3
        self.n_cores = n_cores
        self.tt_rank = tt_rank

        if self.is_naive:
            self.TTqkv = TTLinearSet(in_features=self.dim,
                            out_features=self.dim, n_gates=self.n_gate,
                            bias=qkv_bias, auto_shapes=True,
                            d=self.n_cores, tt_rank=self.tt_rank)
        else:
            shape = tt_shape(self.dim, self.dim,
                             self.n_cores, self.n_gate, new_core=None)
            self.TTqkv = TTLinear(out_features=self.n_gate*self.dim, shape=shape, 
                            bias=qkv_bias, auto_shapes=False, d=self.n_cores,
                            tt_rank=self.tt_rank)
         
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1: # the scale factor in  PVT
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape

        if self.is_tt:
            x = x.reshape(B * N, C)
            qkv = self.TTqkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 3, B, num_head, N, C_head
            q, k, v = qkv[0], qkv[1], qkv[2]  # B, num_head, N, C_head
        else:
            q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B, HW/P^2, HW, C_head
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_is_naive=False,  attn_is_tt=True,
                 mlp_is_naive=False, mlp_is_tt=True,
                 sr_ratio=1, n_cores=4, tt_rank=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio,
            is_naive=attn_is_naive, is_tt=attn_is_tt,
            n_cores=n_cores, tt_rank=tt_rank)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
            is_naive=mlp_is_naive, is_tt=mlp_is_tt,
            n_cores=n_cores, tt_rank=tt_rank)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 attn_is_naive=[False, False, False, False], attn_is_tt=[True, True, True, True],
                 mlp_is_naive=[False, False, False, False], mlp_is_tt=[True, True, True, True],
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
                 n_cores=[4, 4, 4, 4], tt_rank=[8, 8, 8, 8]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0]) # from 224x224x3 to 56x56x64
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1]) # from 56x56x64 to 28x28x128
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2]) # from 28x28x128 to 14x14x320
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3]) # from 14x14x320 to 7x7x512

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            attn_is_naive=attn_is_naive[0], attn_is_tt=attn_is_tt[0],
            mlp_is_naive=mlp_is_naive[0], mlp_is_tt=mlp_is_tt[0],
            sr_ratio=sr_ratios[0], n_cores=n_cores[0], tt_rank=tt_rank[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            attn_is_naive=attn_is_naive[1], attn_is_tt=attn_is_tt[1],
            mlp_is_naive=mlp_is_naive[1], mlp_is_tt=mlp_is_tt[1],
            sr_ratio=sr_ratios[1], n_cores=n_cores[1], tt_rank=tt_rank[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            attn_is_naive=attn_is_naive[2], attn_is_tt=attn_is_tt[2],
            mlp_is_naive=mlp_is_naive[2], mlp_is_tt=mlp_is_tt[2],
            sr_ratio=sr_ratios[2], n_cores=n_cores[2], tt_rank=tt_rank[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            attn_is_naive=attn_is_naive[3], attn_is_tt=attn_is_tt[3],
            mlp_is_naive=mlp_is_naive[3], mlp_is_tt=mlp_is_tt[3],
            sr_ratio=sr_ratios[3], n_cores=n_cores[3], tt_rank=tt_rank[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        # x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x



@BACKBONES.register_module()
class ten_b0(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b0, self).__init__(
            patch_size=4, embed_dims=[32, 64, 160, 256], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)

@BACKBONES.register_module()
class ten_b1_ttall(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b1_ttall, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            attn_is_naive=[False, False, False, False], attn_is_tt=[True, True, True, True],
            mlp_is_naive=[False, False, False, False], mlp_is_tt=[True, True, True, True],
            n_cores=[5, 5, 5, 5], tt_rank=[8, 8, 8, 8],
            drop_rate=0.0, drop_path_rate=0.1)

@BACKBONES.register_module()
class ten_b1_ttall_4cores(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b1_ttall_4cores, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            attn_is_naive=[False, False, False, False], attn_is_tt=[True, True, True, True],
            mlp_is_naive=[False, False, False, False], mlp_is_tt=[True, True, True, True],
            n_cores=[4, 4, 4, 4], tt_rank=[8, 8, 8, 8],
            drop_rate=0.0, drop_path_rate=0.1)

@BACKBONES.register_module()
class ten_b1_ttall_5cores_6rank(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b1_ttall_5cores_6rank, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            attn_is_naive=[False, False, False, False], attn_is_tt=[True, True, True, True],
            mlp_is_naive=[False, False, False, False], mlp_is_tt=[True, True, True, True],
            n_cores=[5, 5, 5, 5], tt_rank=[6, 6, 6, 6],
            drop_rate=0.0, drop_path_rate=0.1)

@BACKBONES.register_module()
class ten_b1_ttmlp(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b1_ttmlp, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            attn_is_naive=[False, False, False, False], attn_is_tt=[False, False, False, False],
            mlp_is_naive=[False, False, False, False], mlp_is_tt=[True, True, True, True],
            n_cores=[5, 5, 5, 5], tt_rank=[8, 8, 8, 8],
            drop_rate=0.0, drop_path_rate=0.1)

@BACKBONES.register_module()
class ten_b1_ttatten(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b1_ttatten, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            attn_is_naive=[False, False, False, False], attn_is_tt=[True, True, True, True],
            mlp_is_naive=[False, False, False, False], mlp_is_tt=[False, False, False, False],
            n_cores=[4, 4, 4, 4], tt_rank=[8, 8, 8, 8],
            drop_rate=0.0, drop_path_rate=0.1)

@BACKBONES.register_module()
class ten_b1_origin(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b1_origin, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1],
            attn_is_naive=[False, False, False, False], attn_is_tt=[False, False, False, False],
            mlp_is_naive=[False, False, False, False], mlp_is_tt=[False, False, False, False],
            n_cores=[4, 4, 4, 4], tt_rank=[8, 8, 8, 8],
            drop_rate=0.0, drop_path_rate=0.1)

@BACKBONES.register_module()
class ten_b2(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b2, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class ten_b3(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b3, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class ten_b4(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b4, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class ten_b5(MixVisionTransformer):
    def __init__(self, **kwargs):
        super(ten_b5, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 6, 40, 3], sr_ratios=[8, 4, 2, 1],
            drop_rate=0.0, drop_path_rate=0.1)