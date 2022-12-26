# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import numpy as np
import torch.nn as nn
import torch
from mmcv.cnn import ConvModule, DepthwiseSeparableConvModule
from collections import OrderedDict
from mmseg.t3nsor.layers import TTLinear
from mmseg.tensorized_rnn.lstm import LSTMCell, LSTM
from mmseg.tensorized_rnn.tt_linearset import TTLinearSet
from mmseg.tensorized_rnn.rnn_utils import tt_shape
from mmseg.ops import resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead
from mmseg.models.utils import *
import attr

from IPython import embed

class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768, n_cores=5, n_gate=1, n_rank=8):
        super().__init__()
        shape_fc1 = tt_shape(input_dim, embed_dim,
                         n_cores, n_gate, new_core=None)
        self.TTfc1 = TTLinear(out_features=embed_dim, shape=shape_fc1,
                              bias=True, auto_shapes=False, d=n_cores,
                              tt_rank=n_rank)
    def forward(self, x):
        # print(f'before: {x.shape}')
        x = x.flatten(2).transpose(1, 2)
        # print(f'after: {x.shape}')
        B, N, C = x.shape
        x=x.contiguous().view(B*N, C)
        x = self.TTfc1(x)
        x = x.contiguous().view(B, N, -1)
        return x


@HEADS.register_module()
class SegFormerHead_TT(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, feature_strides, **kwargs):
        super(SegFormerHead_TT, self).__init__(input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = self.in_channels

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        # n_cores = decoder_params['n_cores']
        # n_rank = decoder_params['n_rank']
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim, n_cores=5, n_rank=6)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim, n_cores=5, n_rank=6)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim, n_cores=5, n_rank=6)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim, n_cores=5, n_rank=6)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim*4,
            out_channels=embedding_dim,
            kernel_size=1,
            norm_cfg=dict(type='SyncBN', requires_grad=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = self._transform_inputs(inputs)  # len=4, 1/4,1/8,1/16,1/32
        c1, c2, c3, c4 = x

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = resize(_c4, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = resize(_c3, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = resize(_c2, size=c1.size()[2:],mode='bilinear',align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
