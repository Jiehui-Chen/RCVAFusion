# Copyright (c) OpenMMLab. All rights reserved.
from typing import Callable, List, Union
import os
import numpy as np
from os import path as osp
from mmdet3d.registry import DATASETS
from mmdet3d.structures import CameraInstance3DBoxes
from mmdet3d.datasets.det3d_dataset import Det3DDataset


@DATASETS.register_module()
class VODDataset(Det3DDataset):
    r"""KITTI Dataset.

    This class serves as the API for experiments on the `KITTI Dataset
    <http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d>`_.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        pipeline (List[dict]): Pipeline used for data processing.
            Defaults to [].
        modality (dict): Modality to specify the sensor data used as input.
            Defaults to dict(use_lidar=True).
        default_cam_key (str): The default camera name adopted.
            Defaults to 'CAM2'.
        load_type (str): Type of loading mode. Defaults to 'frame_based'.

            - 'frame_based': Load all of the instances in the frame.
            - 'mv_image_based': Load all of the instances in the frame and need
              to convert to the FOV-based data type to support image-based
              detector.
            - 'fov_image_based': Only load the instances inside the default
              cam, and need to convert to the FOV-based data type to support
              image-based detector.
        box_type_3d (str): Type of 3D box of this dataset.
            Based on the `box_type_3d`, the dataset will encapsulate the box
            to its original format then converted them to `box_type_3d`.
            Defaults to 'LiDAR' in this dataset. Available options includes:

            - 'LiDAR': Box in LiDAR coordinates.
            - 'Depth': Box in depth coordinates, usually for indoor dataset.
            - 'Camera': Box in camera coordinates.
        filter_empty_gt (bool): Whether to filter the data with empty GT.
            If it's set to be True, the example with empty annotations after
            data pipeline will be dropped and a random example will be chosen
            in `__getitem__`. Defaults to True.
        test_mode (bool): Whether the dataset is in test mode.
            Defaults to False.
        pcd_limit_range (List[float]): The range of point cloud used to filter
            invalid predicted boxes.
            Defaults to [0, -40, -3, 70.4, 40, 0.0].
    """
    METAINFO = {
        'classes': ('Car', 'Pedestrian', 'Cyclist', 'rider', 'truck', 'bicycle', 'motor', 'moped_scooter', 'ride_other',
                    'bicycle_rack', 'human_depiction', 'vehicle_other', 'ride_uncertain', 'DontCare'),
    }

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Union[dict, Callable]] = [],
                 data_prefix: dict = dict(pts_lidar='lidar/training/velodyne', img='lidar/training/image_2',pts_radar='radar/training/velodyne'),
                 modality: dict = dict(use_lidar=True,use_radar=False,use_camera=True),
                 point_type: str ='use_radar',
                 default_cam_key: str = 'CAM2',
                 load_type: str = 'frame_based',
                 box_type_3d: str = 'LiDAR',
                 filter_empty_gt: bool = True,
                 test_mode: bool = False,
                 pcd_limit_range: List[float] = [0, -40, -3, 70.4, 40, 0.0],
                 **kwargs) -> None:

        self.pcd_limit_range = pcd_limit_range
        assert load_type in ('frame_based', 'mv_image_based',
                             'fov_image_based')


        self.load_type = load_type
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            data_prefix=data_prefix,
            modality=modality,
            default_cam_key=default_cam_key,
            box_type_3d=box_type_3d,
            filter_empty_gt=filter_empty_gt,
            test_mode=test_mode,
            **kwargs)
        assert self.modality is not None
        assert box_type_3d.lower() in ('lidar', 'camera')
    def dynamic_baseline(self, info):
        P2 = np.array(np.array(info['images']['CAM2']['cam2img']).astype(np.float32)) # cam2img
        P3 = np.array(np.array(info['images']['CAM3']['cam2img']).astype(np.float32)) # cam2img
        baseline = P3[0,3]/(-P3[0,0]) - P2[0,3]/(-P2[0,0])
        return baseline

    def parse_data_info(self, info: dict) -> dict:
        """Process the raw data info.

        The only difference with it in `Det3DDataset`
        is the specific process for `plane`.

        Args:
            info (dict): Raw info dict.

        Returns:
            dict: Has `ann_info` in training stage. And
            all path has been converted to absolute path.
        """
        if self.modality['use_lidar']:


            info['plane'] = None

        # if self.load_type == 'fov_image_based' and self.load_eval_anns:
        #     info['instances'] = info['cam_instances'][self.default_cam_key]

        # info = super().parse_data_info(info)

        if self.modality['use_lidar']:
            info['lidar_points']['lidar_path'] = \
                osp.join(
                    self.data_prefix.get('pts_lidar', ''),
                    info['lidar_points']['lidar_path'])

            info['num_pts_feats'] = info['lidar_points']['num_pts_feats']
            info['lidar_path'] = info['lidar_points']['lidar_path']

            if 'lidar_sweeps' in info:
                for sweep in info['lidar_sweeps']:
                    file_suffix = sweep['lidar_points']['lidar_path'].split(
                        os.sep)[-1]
                    if 'samples' in sweep['lidar_points']['lidar_path']:
                        sweep['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix['pts'], file_suffix)
                    else:
                        sweep['lidar_points']['lidar_path'] = osp.join(
                            self.data_prefix['sweeps'], file_suffix)


        if self.modality['use_radar']:
            info['radar_points']['radar_path'] = \
                osp.join(
                    self.data_prefix.get('pts_radar', ''),
                    info['radar_points']['radar_path'])

            info['num_pts_feats'] = info['radar_points']['num_pts_feats']
            info['radar_path'] = info['radar_points']['radar_path']
            # info['lidar2img'] = np.array(info['radar_points']['radar2img'])
            info['lidar2cam'] = np.array(info['radar_points']['radar2cam'])
            info['cam2img'] = np.array(info['images']['CAM2']['cam2img'])
            info['focal_length'] = info['cam2img'][0][0]
            info['baseline'] = self.dynamic_baseline(info)

        if self.modality['use_camera']:

            img_filename = info['images']['CAM2']['img_path']
            info['img_path'] = os.path.join(self.data_prefix['img'], img_filename)


        if not self.test_mode:
            # used in training
            info['ann_info'] = self.parse_ann_info(info)
        if self.test_mode and self.load_eval_anns:
            info['eval_ann_info'] = self.parse_ann_info(info)

        return info

    def parse_ann_info(self, info: dict) -> dict:
        """Process the `instances` in data info to `ann_info`.

        Args:
            info (dict): Data information of single data sample.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`):
                  3D ground truth bboxes.
                - bbox_labels_3d (np.ndarray): Labels of ground truths.
                - gt_bboxes (np.ndarray): 2D ground truth bboxes.
                - gt_labels (np.ndarray): Labels of ground truths.
                - difficulty (int): Difficulty defined by KITTI.
                  0, 1, 2 represent xxxxx respectively.
        """
        ann_info = super().parse_ann_info(info)
        if ann_info is None:
            ann_info = dict()
            # empty instance
            ann_info['gt_bboxes_3d'] = np.zeros((0, 7), dtype=np.float32)
            ann_info['gt_labels_3d'] = np.zeros(0, dtype=np.int64)

            if self.load_type in ['fov_image_based', 'mv_image_based']:
                ann_info['gt_bboxes'] = np.zeros((0, 4), dtype=np.float32)
                ann_info['gt_bboxes_labels'] = np.array(0, dtype=np.int64)
                ann_info['centers_2d'] = np.zeros((0, 2), dtype=np.float32)
                ann_info['depths'] = np.zeros((0), dtype=np.float32)

        ann_info = self._remove_dontcare(ann_info)
        # in kitti, lidar2cam = R0_rect @ Tr_velo_to_cam
        lidar2cam = np.array(info['lidar_points']['lidar2cam'])
        radar2cam=np.array(info['radar_points']['radar2cam'])
        if self.modality['use_lidar']:
            # convert gt_bboxes_3d to velodyne coordinates with `lidar2cam`
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d,
                                                     np.linalg.inv(lidar2cam))
        elif self.modality['use_radar']:
            gt_bboxes_3d = CameraInstance3DBoxes(
                ann_info['gt_bboxes_3d']).convert_to(self.box_mode_3d,
                                                     np.linalg.inv(radar2cam))
        ann_info['gt_bboxes_3d'] = gt_bboxes_3d
        return ann_info
