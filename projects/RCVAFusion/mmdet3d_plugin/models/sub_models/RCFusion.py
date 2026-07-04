import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmdet3d.registry import MODELS

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        output = self.sigmoid(output)
        return output

@MODELS.register_module()
class DFA(nn.Module):
    def __init__(self, img_channels=256, rad_channels=256, association_channels=256, out_channels=256, padding=1):
        super(DFA, self).__init__()
        self.fusion_channels = out_channels

        self.ca = ChannelAttention(channel=self.fusion_channels,reduction=8)
        self.ca_img = ChannelAttention(channel=img_channels,reduction=8)
        # self.sa = SpatialAttention(kernel_size=7)
        self.radar_img_association_fusion = nn.Sequential(
            nn.Conv2d(association_channels+rad_channels, self.fusion_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.fusion_channels),
            nn.ReLU(inplace=True)
        )

        self.pred_occupancy = nn.Sequential(
            nn.Conv2d(self.fusion_channels,
                      self.fusion_channels // 2,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='zeros'),
            nn.BatchNorm2d(self.fusion_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fusion_channels // 2,
                      1,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Sigmoid()
        )
        self.pred_occupancy = nn.Sequential(
            nn.Conv2d(self.fusion_channels,
                      self.fusion_channels,
                      kernel_size=1,
                      stride=1,
                      bias=False,
                      padding=0,
                      dilation=1),
            nn.BatchNorm2d(self.fusion_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fusion_channels,
                      self.fusion_channels,
                      kernel_size=3,
                      stride=1,
                      bias=False,
                      padding=6,
                      dilation=6),
            nn.BatchNorm2d(self.fusion_channels),
            nn.ReLU(inplace=True),
            *self.pred_occupancy)
        self.reduce_mixBEV = ConvModule(
            self.fusion_channels*2,
            self.fusion_channels,
            3,
            padding=1,
            conv_cfg=None,
            norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
            act_cfg=dict(type='ReLU'),
            inplace=False)

    def forward(self,radar_feats,association_feats,img_bev_feats):
        #first step
        x = self.radar_img_association_fusion(torch.cat([radar_feats, association_feats], dim=1))

        x = x * self.ca(x)
        # x = x * self.sa(x)
        radar_association_fusion = x

        #second step
        img_bev_feats=img_bev_feats*self.ca_img(img_bev_feats)
        radar_occ = self.pred_occupancy(radar_association_fusion)
        img_bev_feature = img_bev_feats * radar_occ
        radar_association_fusion = radar_association_fusion * radar_occ
        fusion_bev = self.reduce_mixBEV(torch.cat([radar_association_fusion, img_bev_feature], dim=1))
        return fusion_bev
