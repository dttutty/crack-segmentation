# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch.nn as nn
from mmcv.cnn import ConvModule

from mmseg.ops import Upsample, resize
from ..builder import HEADS
from .decode_head import BaseDecodeHead


@HEADS.register_module()
class FPNHead(BaseDecodeHead):
    """Panoptic Feature Pyramid Networks.

    This head is the implementation of `Semantic FPN
    <https://arxiv.org/abs/1901.02446>`_.

    Args:
        feature_strides (tuple[int]): The strides for input feature maps.
            stack_lateral. All strides suppose to be power of 2. The first
            one is of largest resolution.
    """

    def __init__(self, feature_strides, **kwargs):
        super(FPNHead, self).__init__(
            input_transform='multiple_select', **kwargs)
        assert len(feature_strides) == len(self.in_channels)
        assert min(feature_strides) == feature_strides[0]
        self.feature_strides = feature_strides

        self.scale_heads = nn.ModuleList()
        #while feature_strides = [4,8,16,32]
        #i=0, feature_stride[0] = 4
        #i=1, feature_stride[1] = 8
        #i=2, feature_stride[2] = 16
        #i=2, feature_stride[3] = 32
        for i in range(len(feature_strides)):
            head_length = max(
                1,
                int(np.log2(feature_strides[i]) - np.log2(feature_strides[0])))
            scale_head = []
            for k in range(head_length):
                #feature_stride[0] = 4, head_length=1, k=0, scale_head.append(conv(self.in_channels[0], channels))
                
                #feature_stride[1] = 8, head_length=1, k=0, scale_head.append(conv(self.in_channels[1], channels))
                #append(Upsample())
                
                #feature_stride[2] = 16, head_length=2, k=0, scale_head.append(conv(self.in_channels[2], channels))
                #feature_stride[2] = 16, head_length=2, k=1, scale_head.append(conv(channels, channels))
                #append(Upsample())
                
                #feature_stride[3] = 32, head_length=2, k=0, scale_head.append(conv(self.in_channels[3], channels))
                #feature_stride[3] = 32, head_length=2, k=1, scale_head.append(conv(channels, channels))
                #feature_stride[3] = 32, head_length=2, k=2, scale_head.append(conv(channels, channels))
                #append(Upsample())
                scale_head.append(
                    ConvModule(
                        self.in_channels[i] if k == 0 else self.channels,
                        self.channels,
                        3,
                        padding=1,
                        conv_cfg=self.conv_cfg,
                        norm_cfg=self.norm_cfg,
                        act_cfg=self.act_cfg))
                if feature_strides[i] != feature_strides[0]: #feature_stride[0] = 4
                    scale_head.append(
                        Upsample(
                            scale_factor=2,
                            mode='bilinear',
                            align_corners=self.align_corners))
            self.scale_heads.append(nn.Sequential(*scale_head))

    def forward(self, inputs):

        x = self._transform_inputs(inputs)

        output = self.scale_heads[0](x[0])
        for i in range(1, len(self.feature_strides)):
            # non inplace
            output = output + resize(
                self.scale_heads[i](x[i]),
                size=output.shape[2:],
                mode='bilinear',
                align_corners=self.align_corners)

        output = self.cls_seg(output)
        return output
