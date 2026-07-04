import time

import torch,copy
from copy import deepcopy
from mmdet.structures import SampleList
from torch import Tensor
import torch.nn as nn
import numpy as np
from mmdet3d.registry import MODELS
from mmdet3d.utils import ConfigType, OptConfigType, OptMultiConfig, OptSampleList
from mmdet3d.models import Base3DDetector
from typing import Tuple, Union
from typing import Dict, List, Optional, Tuple
from torch.nn import functional as F
from mmdet3d.models.data_preprocessors.voxelize import VoxelizationByGridShape

import matplotlib.pyplot as plt
import os
from torchvision.utils import save_image

@MODELS.register_module()
class RCVAFusion_Detector(Base3DDetector):
    def __init__(self,
                 freeze_images=True,
                 downsample=None,
                 data_preprocessor: OptConfigType = None,
                 img_backbone: Optional[dict] = None,
                 img_neck: Optional[dict] = None,
                 view_transformer: Optional[dict] = None,
                 img_sampling_net: Optional[dict] = None,
                 pts_voxel_encoder: Optional[dict] =None,
                 pts_middle_encoder: Optional[dict] =None,
                 img_voxel_layer:  Optional[dict] =None,
                 pts_backbone: Optional[dict] =None,
                 pts_neck: Optional[dict] =None,
                 projection_voxel_layer: Optional[dict] = None,
                 projection_voxel_encoder: Optional[dict] = None,
                 projection_middle_encoder: Optional[dict] = None,
                 projection_backbone: Optional[dict] = None,
                 projection_neck: Optional[dict] = None,
                 fusion_layer: Optional[dict] = None,
                 bbox_head: Optional[dict] =None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 enable_draw_bev_feature_map: bool = False,
                 enable_recording_fps: bool = False,
                 init_cfg: OptMultiConfig = None,):
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.downsample = downsample
        self.projection_backbone = MODELS.build(projection_backbone)
        self.projection_middle_encoder = MODELS.build(projection_middle_encoder)
        self.projection_voxel_encoder = MODELS.build(projection_voxel_encoder)
        self.projection_neck = MODELS.build(projection_neck)
        self.view_transform = MODELS.build(view_transformer)
        self.img_voxel_layer = VoxelizationByGridShape(**img_voxel_layer)
        self.projection_voxel_layer = VoxelizationByGridShape(**projection_voxel_layer)
        self.img_sampling_net = MODELS.build(img_sampling_net)
        self.pts_voxel_encoder = MODELS.build(pts_voxel_encoder)
        self.pts_middle_encoder = MODELS.build(pts_middle_encoder)
        self.pts_backbone = MODELS.build(pts_backbone)
        self.pts_neck = MODELS.build(pts_neck)
        self.fusion_modality = MODELS.build(fusion_layer)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.img_backbone = MODELS.build(
            img_backbone) if img_backbone is not None else None

        self.img_neck = MODELS.build(img_neck)

        self.conv_reduce = nn.Sequential(
            nn.Conv2d(1280, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.rad_channels=256
        self.radar_fusion = nn.Sequential(
            nn.Conv2d(256+256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.freeze_images = freeze_images
        self.init_weights()
        if self.freeze_images: self.freeze_img_model()
        self.enable_draw_bev_feature_map = enable_draw_bev_feature_map
        self.enable_recording_fps = enable_recording_fps
        self.record_fps = {'num': 0, 'time': 0}

    def freeze_img_model(self):
        """freeze image backbone and neck for fusion"""

        for param in self.img_backbone.parameters():
            param.requires_grad = False

        for param in self.img_neck.parameters():
            param.requires_grad = False

    def extract_img_feat(self, img, batch_data_metas):
        N = 1
        h, w = img.shape[2] // self.downsample, img.shape[3] // self.downsample
        x = self.img_backbone(img)

        x = self.img_neck(x)
        align_feats = [F.interpolate(feat, (h, w), mode='bilinear', align_corners=True) for feat in x]
        align_feats = torch.cat(align_feats, dim=1)
        align_feats = self.conv_reduce(align_feats)

        # B, C, H, W = align_feats.shape

        return align_feats

    def extract_pts_feat(self, batch_inputs_dict):
        """Extract features from points."""
        voxel_dict = batch_inputs_dict['voxels']
        voxel_features = self.pts_voxel_encoder(voxel_dict['voxels'],
                                                voxel_dict['num_points'],
                                                voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, voxel_dict['coors'],
                                    batch_size)

        x = self.pts_backbone(x)
        x = self.pts_neck(x)

        return x[0]

    def extract_projection_feat(self,projection_points):

        voxel_dict = self.voxelize_2(projection_points)
        voxel_features = self.projection_voxel_encoder(voxel_dict['voxels'],
                                            voxel_dict['num_points'],
                                            voxel_dict['coors'])
        batch_size = voxel_dict['coors'][-1, 0].item() + 1
        x = self.projection_middle_encoder(voxel_features, voxel_dict['coors'],
                                batch_size)
        x = self.projection_backbone(x)

        x = self.projection_neck(x)

        return x[-1]

    def extract_feat(self, batch_inputs_dict: dict, batch_data_metas):
        """Extract features from points."""
        start_time=time.time()
        img = batch_inputs_dict.get('imgs', None)
        voxel_dict = batch_inputs_dict['voxels']
        points = batch_inputs_dict['points']
        virtual_points = self.get_virtul_points(points)

        voxel_dict_img = self.voxelize(virtual_points)

        batch_size = voxel_dict['coors'][-1, 0].item() + 1

        cam_aware, bda_rot, lidar_aug_matrix, img_aug_matrix, projection_points = self.preprocessing_information(
            batch_data_metas, img.device)


        projection_bev = self.extract_projection_feat(projection_points)
        batch_img_metas = self.reorganize_lidar2img(batch_data_metas)
        calib = []
        for sample_idx in range(batch_size):
            mat = batch_img_metas[sample_idx]['final_lidar2img']
            mat = torch.Tensor(mat).to(img.device)
            calib.append(mat)
        final_lidar2img = torch.stack(calib)
        pts_bev_feat = self.extract_pts_feat(batch_inputs_dict)

        img_feature = self.extract_img_feat(img, batch_data_metas)

        img_OAS_feat = self.img_sampling_net(voxel_dict_img['coors'], batch_size, final_lidar2img, img_feature)
        img_inputs = [cam_aware[0], cam_aware[1], cam_aware[2], cam_aware[3], cam_aware[4], bda_rot]
        rots, trans, intrins, post_rots, post_trans, bda = img_inputs[0:6]
        view_trans_inputs = [rots, trans, intrins, post_rots, post_trans, bda]

        B, C, H, W = img_feature.shape
        img_feature = img_feature.view(B, 1, -1, H, W)

        img_VRPDL_feat, img_bev_feat = self.view_transform(img_feature, projection_bev, view_trans_inputs)

        img_bev_feat = img_bev_feat.mean(-1)  # B, C, bev_h_, bev_w_
        img_bev_feat = img_bev_feat.permute(0, 1, 3, 2).contiguous()
        img_VRPDL_feat =  img_VRPDL_feat.mean(-1)
        img_VRPDL_feat =  img_VRPDL_feat.permute(0,1,3,2).contiguous()


        association_feat = self.radar_fusion(torch.cat([ img_VRPDL_feat, img_OAS_feat], dim=1))


        x = self.fusion_modality(pts_bev_feat,association_feat,img_bev_feat)
        end_time=time.time()
        if not self.training and self.enable_draw_bev_feature_map:
            self.draw_bev_feature_map(x, batch_data_metas)
        if not self.training and self.enable_recording_fps:
            self.recording_fps(end_time - start_time)
        return [x]
    def recording_fps(self, step_all_time):
        self.record_fps['num'] += 1
        self.record_fps['time'] += step_all_time
        if not self.training and self.record_fps['num'] % 50 == 0:
            print(' FPS: %.2f'%(1.0/step_all_time))
        if not self.training and self.record_fps['num'] == 1296 and not self.training:
            print(' FINAL VOD FPS: %.2f'%(1296/self.record_fps['time']))

    def get_virtul_points(self, points):
        for i,cur_points in enumerate(points):
            virtual_points = cur_points[cur_points[:,-2]==1]
            if len(virtual_points)>0:
                points[i]=virtual_points
        return points

    def reorganize_lidar2img(self, batch_input_metas):
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for img_metas in batch_input_metas:
            final_cam2img = copy.deepcopy(img_metas['cam2img'])


            # same as visualization in BEVAug3D
            rots, trans, intrins, post_rots, post_trans = img_metas['cam_aware'][:5]
            final_cam2img[:2, :3] = post_rots[:2, :2] @ final_cam2img[:2, :3]
            final_cam2img[:2, 2] = post_trans[:2] + final_cam2img[:2, 2]
            final_lidar2img = final_cam2img @ img_metas['lidar2cam']
            final_lidar2img = final_lidar2img @ np.linalg.inv(img_metas['lidar_aug_matrix'])
            img_metas['final_lidar2img'] = final_lidar2img


        return batch_input_metas

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        x = self.extract_feat(batch_inputs_dict,batch_input_metas)
        losses = self.bbox_head.loss(x, batch_data_samples, **kwargs)
        return losses


    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        batch_input_metas = [item.metainfo for item in batch_data_samples]
        x = self.extract_feat(batch_inputs_dict, batch_input_metas)
        results_list = self.bbox_head.predict(x, batch_data_samples, **kwargs)
        predictions = self.add_pred_to_datasample(batch_data_samples,
                                                  results_list)
        return predictions

    def _forward(self,
                 batch_inputs: Tensor,
                 batch_data_samples: OptSampleList = None):
        """Network forward process.

        Usually includes backbone, neck and head forward without any post-
        processing.
        """
        pass

    def preprocessing_information(self, batch_img_metas, device):
        if self.training:
            cam_aware = [img_meta['cam_aware'] for img_meta in batch_img_metas]
            merged_tensors = [None] * len(cam_aware[0])
            for i in range(len(cam_aware[0])):
                component = [x[i] for x in cam_aware]
                merged_tensors[i] = torch.stack(component, dim=0)
            cam_aware = merged_tensors
            cam_aware = [x.to(device) for x in cam_aware]
            # img_aug_matrix: 4x4 martix of combined post_rot&post_tran of IMG_AUG
            img_aug_matrix = [img_meta['img_aug_matrix'] for img_meta in batch_img_metas]
            img_aug_matrix = torch.tensor(np.stack(img_aug_matrix, axis=0))
            img_aug_matrix = img_aug_matrix.to(device)

            projection_points = [img_meta['projection_points'].tensor.to(device) for img_meta in batch_img_metas]

            if 'lidar_aug_matrix' in batch_img_metas[0]:
                lidar_aug_matrix = [img_meta['lidar_aug_matrix'] for img_meta in batch_img_metas]
                lidar_aug_matrix = torch.tensor(np.stack(lidar_aug_matrix, axis=0)).to(torch.float32)
                bda_rot = [img_meta['bda_rot'] for img_meta in batch_img_metas]
                bda_rot = torch.tensor(np.stack(bda_rot, axis=0)).to(torch.float32)
            else:
                lidar_aug_matrix = torch.eye(4).unsqueeze(0).repeat(len(batch_img_metas), 1, 1)
                bda_rot = lidar_aug_matrix
            lidar_aug_matrix = lidar_aug_matrix.to(device)
            bda_rot = bda_rot.to(device)


        else:
            batch_img_metas = batch_img_metas[0]
            cam_aware = batch_img_metas['cam_aware']
            cam_aware = [[x.to(device)] for x in cam_aware]
            cam_aware = [torch.stack(x, dim=0) for x in cam_aware]
            if 'projection_points' in batch_img_metas:
                projection_points = [batch_img_metas['projection_points'].tensor.to(device)]

            img_aug_matrix = [batch_img_metas['img_aug_matrix']]
            img_aug_matrix = torch.tensor(np.stack(img_aug_matrix, axis=0))
            img_aug_matrix = img_aug_matrix.to(device)

            if 'lidar_aug_matrix' in batch_img_metas:
                lidar_aug_matrix = [batch_img_metas['lidar_aug_matrix']]
                lidar_aug_matrix = torch.tensor(np.stack(lidar_aug_matrix, axis=0)).to(torch.float32)
                bda_rot = [batch_img_metas['bda_rot'] ]
                bda_rot = torch.tensor(np.stack(bda_rot, axis=0)).to(torch.float32)
            else:
                lidar_aug_matrix = torch.eye(4).unsqueeze(0)
                bda_rot = lidar_aug_matrix
            lidar_aug_matrix = lidar_aug_matrix.to(device)
            bda_rot = bda_rot.to(device)


        return cam_aware,bda_rot,lidar_aug_matrix,img_aug_matrix,projection_points

    @torch.no_grad()
    def voxelize(self, points: List[Tensor]) -> Dict[str, Tensor]:
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()


        voxels, coors, num_points, voxel_centers = [], [], [], []
        for i, res in enumerate(points):
            res_voxels, res_coors, res_num_points = self.img_voxel_layer(res)
            res_voxel_centers = (
                res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                    self.img_voxel_layer.voxel_size) + res_voxels.new_tensor(
                        self.img_voxel_layer.point_cloud_range[0:3])
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            voxel_centers.append(res_voxel_centers)

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        num_points = torch.cat(num_points, dim=0)
        voxel_centers = torch.cat(voxel_centers, dim=0)

        voxel_dict['num_points'] = num_points
        voxel_dict['voxel_centers'] = voxel_centers


        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict

    @torch.no_grad()
    def voxelize_2(self, points: List[Tensor]) -> Dict[str, Tensor]:
        """Apply voxelization to point cloud.

        Args:
            points (List[Tensor]): Point cloud in one data batch.
            data_samples: (list[:obj:`Det3DDataSample`]): The annotation data
                of every samples. Add voxel-wise annotation for segmentation.

        Returns:
            Dict[str, Tensor]: Voxelization information.

            - voxels (Tensor): Features of voxels, shape is MxNxC for hard
              voxelization, NxC for dynamic voxelization.
            - coors (Tensor): Coordinates of voxels, shape is Nx(1+NDim),
              where 1 represents the batch index.
            - num_points (Tensor, optional): Number of points in each voxel.
            - voxel_centers (Tensor, optional): Centers of voxels.
        """

        voxel_dict = dict()


        voxels, coors, num_points, voxel_centers = [], [], [], []
        for i, res in enumerate(points):
            res_voxels, res_coors, res_num_points = self.projection_voxel_layer(res)
            res_voxel_centers = (
                res_coors[:, [2, 1, 0]] + 0.5) * res_voxels.new_tensor(
                    self.projection_voxel_layer.voxel_size) + res_voxels.new_tensor(
                        self.projection_voxel_layer.point_cloud_range[0:3])
            res_coors = F.pad(res_coors, (1, 0), mode='constant', value=i)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
            voxel_centers.append(res_voxel_centers)

        voxels = torch.cat(voxels, dim=0)
        coors = torch.cat(coors, dim=0)
        num_points = torch.cat(num_points, dim=0)
        voxel_centers = torch.cat(voxel_centers, dim=0)

        voxel_dict['num_points'] = num_points
        voxel_dict['voxel_centers'] = voxel_centers


        voxel_dict['voxels'] = voxels
        voxel_dict['coors'] = coors

        return voxel_dict
    def draw_bev_feature_map(self, bev_feats, img_metas, bev_feats_name='bev_feats_fusion'):

        figures_path_bevnd = "/home/chenjh/chenjh/dataset/Data/data_with_create/association_result/bev_tj4d_view"

        b, _, h, w = bev_feats.shape
        # bev_feats = bev_feats.mean(1).unsqueeze(1) # using mean
        bev_feats_show = bev_feats.max(1, keepdim=True).values  # using max
        # bev_feats_show = torch.rot90(bev_feats_show, k=2, dims=(2, 3))\
        bev_feats_show = torch.flip(bev_feats_show, [2])  # horizontal flip for consistency to gt bev bbox
        for i in range(bev_feats.shape[0]):
            img_name = img_metas[i]['img_path'].split('/')[-1].split('.')[0]
            bev_feats_tmp = bev_feats_show[i:i + 1, :, :, :]
            bev_feats_tmp = (bev_feats_tmp - bev_feats_tmp.min()) / (bev_feats_tmp.max() - bev_feats_tmp.min())
            # bev_feats_tmp = (bev_feats_tmp - 0.75)/(1.00 - 0.75)
            if bev_feats_name == 'bev_feats_feats': bev_feats_tmp = bev_feats_tmp * 25
            bev_feats_tmp_np = bev_feats_tmp.squeeze().cpu().detach().numpy()
            bev_feats_tmp_colored = plt.cm.viridis(bev_feats_tmp_np)[..., :3]
            bev_feats_tmp_colored = torch.tensor(bev_feats_tmp_colored).permute(2, 0, 1).unsqueeze(0)
            save_image(bev_feats_tmp_colored, os.path.join(figures_path_bevnd, img_name + '.png'))



