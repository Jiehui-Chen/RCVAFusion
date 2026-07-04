from torch import nn
import torch
import numpy as np
from mmdet3d.registry import MODELS
from typing import Dict, List, Optional, Tuple, Union
from torch import Tensor
from functools import partial
import torch.nn.functional as F
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv


def index2points(indices, pts_range=[0, -39.68, -3, 69.12, 39.68, 2.76], voxel_size=[0.04, 0.04, 0.08], stride=8):
    """
    convert 3D voxel indices to a set of grid points.
    """

    voxel_size = np.array(voxel_size) * stride
    min_x = pts_range[0] + voxel_size[0] / 2
    min_y = pts_range[1] + voxel_size[1] / 2
    min_z = pts_range[2] + voxel_size[2] / 2

    new_indices = indices.clone().float()
    indices_float = indices.clone().float()
    new_indices[:, 1] = indices_float[:, 3] * voxel_size[0] + min_x
    new_indices[:, 2] = indices_float[:, 2] * voxel_size[1] + min_y
    new_indices[:, 3] = indices_float[:, 1] * voxel_size[2] + min_z

    return new_indices
def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m
def post_act_block2d(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                     conv_type='subm', norm_fn=None):
    if conv_type == 'subm':
        conv = spconv.SubMConv2d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
        relu = nn.ReLU()
    elif conv_type == 'spconv':
        conv = spconv.SparseConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
        relu = nn.ReLU(inplace=True)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv2d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
        relu = nn.ReLU()
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        relu,
    )

    return m

def index2uv(indices, batch_size, lidar2img, stride):
    """
    convert the 3D voxel indices to image pixel indices.
    """
    device = indices.device
    new_uv = indices.new(size=(indices.shape[0], 3))
    depth = indices.new(size=(indices.shape[0], 1)).float()
    for b_i in range(batch_size):
        cur_in = indices[indices[:, 0] == b_i]
        cur_pts = index2points(cur_in, stride=stride)
        cur_pts = cur_pts[:, 1:4]
        pts_hom = torch.cat((cur_pts, torch.ones((cur_pts.shape[0], 1), device=device)), dim=1)  # (N, 4)

        pts_img = torch.matmul(lidar2img[b_i], pts_hom.t()).t()  # (N, 4)
        pts_img[:, :2] = pts_img[:, :2] / pts_img[:, 2:3]
        pts_rect_depth = pts_img[:, 2]

        pts_img = pts_img.int()
        # pts_img = torch.from_numpy(pts_img).to(new_uv.device)
        new_uv[indices[:, 0] == b_i, 1:3] = pts_img[:, :2]
        # pts_rect_depth = torch.from_numpy(pts_rect_depth).to(new_uv.device).float()
        depth[indices[:, 0] == b_i, 0] = pts_rect_depth[:]
    new_uv[:, 0] = indices[:, 0]
    new_uv[:, 1] = torch.clamp(new_uv[:, 1], min=0, max=1024 - 1) // stride
    new_uv[:, 2] = torch.clamp(new_uv[:, 2], min=0, max=768 - 1) // stride

    return new_uv, depth
class Basicblock(nn.Module):
    def __init__(self, input_c=16, output_c=16, stride=1, padding=1, indice_key='vir1', conv_depth=False
                 ):
        super().__init__()
        self.stride = stride
        block = post_act_block
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.conv_depth = conv_depth

        if self.stride > 1:
            self.down_layer = block(input_c,
                                    output_c,
                                    3,
                                    norm_fn=norm_fn,
                                    stride=stride,
                                    padding=padding,
                                    indice_key=('sp' + indice_key),
                                    conv_type='spconv')
        c1 = input_c

        if self.stride > 1:
            c1 = output_c
        if self.conv_depth:
            c1 += 4

        c2 = output_c

        self.d3_conv1 = block(c1,
                              c2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm1' + indice_key))
        self.d3_conv2 = block(c2,
                              c2,
                              3,
                              norm_fn=norm_fn,
                              padding=1,
                              indice_key=('subm2' + indice_key))

    def forward(self, sp_tensor):

        if self.stride > 1:
            sp_tensor = self.down_layer(sp_tensor)

        d3_feat1 = self.d3_conv1(sp_tensor)
        d3_feat2 = self.d3_conv2(d3_feat1)
        return d3_feat2
def get_feature_from_img(img_feature, indices, lidar2img, batch_size, stride=8):
    uv_coords, depth = index2uv(indices, batch_size, lidar2img, 1)
    uv_coords[:, 1:] = (uv_coords[:, 1:] / stride).int()
    feature = []
    for b in range(batch_size):
        cur_coords = uv_coords[uv_coords[:, 0] == b]
        cur_coords = cur_coords[:, 1:3]
        cur_img_feature = img_feature[b, :, cur_coords[:,1], cur_coords[:,0]]
        cur_img_feature = cur_img_feature.permute(1, 0)
        feature.append(cur_img_feature)
    feature = torch.cat(feature,dim=0)
    return feature, uv_coords

@MODELS.register_module()
class TJ4D_OAS(nn.Module):
    def __init__(self,):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.img_conv1 = Basicblock(256, 128, stride=1, indice_key='vir_img_1')
        self.img_conv2 = Basicblock(128, 64, stride=1, indice_key='vir_img_2')

        self.img_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=0,
                                bias=False, indice_key='spconv_down_img'),
            norm_fn(64),
            nn.ReLU(),
        )

    def forward(self, coors: Tensor,
                    batch_size: int, final_lidar2img, img_feature) -> Union[Tensor, Tuple[Tensor, list]]:
        coors = coors.int()
        coors_8_down = coors.clone()
        coors_8_down[:, 1:] = (coors_8_down[:, 1:]/8).int()

        img_uv_feature, uv_coords = get_feature_from_img(img_feature, coors, final_lidar2img, batch_size)

        img_sp_tensor = spconv.SparseConvTensor(img_uv_feature, coors_8_down, [9, 248, 216], batch_size)
        img_conv1 = self.img_conv1(img_sp_tensor)
        img_conv2 = self.img_conv2(img_conv1)

        img_out = self.img_out(img_conv2)

        img_spatial_features = img_out.dense()

        N_i, C_i, D_i, H_i, W_i = img_spatial_features.shape

        img_spatial_features = img_spatial_features.view(N_i, C_i * D_i, H_i, W_i)
        # img_spatial_features = F.interpolate(img_spatial_features, (160, 160), mode='bilinear', align_corners=True)

        return img_spatial_features