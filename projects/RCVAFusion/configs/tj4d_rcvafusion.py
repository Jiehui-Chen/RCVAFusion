_base_ = ['../../../configs/_base_/default_runtime.py']

custom_imports = dict(imports=['projects.RCVAFusion.mmdet3d_plugin'])

virtual_points_root_path='data/TJ4DRadSet/tj4d_virtual_points'

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0], to_rgb=False,
)
point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 2.76]
voxel_size = [0.32, 0.32, 5.76]
img_voxel_size=[0.04, 0.04, 0.08]
projection_cloud_range = [0.0, 1.0, 0, 1024, 73.0, 768]
projection_voxel_size = [4, 0.5, 768]
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_size[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_size[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_size[2]],
    'dbound': [1.0, 73, 1.0]}
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675],
    std=[1.0, 1.0, 1.0], to_rgb=False,
)
ida_aug_conf = {
    'resize_lim': (0.70, 0.80),
    'final_dim': (768, 1024),
    'final_dim_test': (768, 1024), # 960 1280
    'bot_pct_lim': (0.0, 0.0),
    'top_pct_lim': (0.0, 0.1),
    'rot_lim': (-2.7, 2.7),
    'rand_flip': True,
}
bda_aug_conf = dict(
    rot_range=(-0.3925, 0.3925),
    scale_ratio_range=(0.95, 1.05),
    translation_std=(1.0, 1.0, 0.0),
    flip_dx_ratio=0.0, # no need for KITTI, which x > 0
    flip_dy_ratio=0.5,
)
downsample = 8
_dim_ = 256
freeze_images = True

model=dict(
    type='RCVAFusion_Detector',
    enable_draw_bev_feature_map=False,
    enable_recording_fps=False,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=12,  # max_points_per_voxel
            point_cloud_range=point_cloud_range,
            voxel_size=[voxel_size[0]/2, voxel_size[1]/2, voxel_size[2]],
            max_voxels=(16000, 40000))),

    downsample=downsample,
    freeze_images = freeze_images,
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        # TODO: figure if requires_grad
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='caffe'),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        # make the image features more stable numerically to avoid loss nan
        norm_cfg=dict(type='BN', requires_grad=False),
        num_outs=5),
    view_transformer=dict(
        type='ViewTransformerLSS_VRPDL',
        downsample=downsample,
        grid_config=grid_config,
        data_config=ida_aug_conf,
        occupancy_height=96),
    img_voxel_layer=dict(
        max_num_points=3,
        point_cloud_range=point_cloud_range,
        voxel_size=img_voxel_size,
        max_voxels=(16000, 40000)
    ),
    img_sampling_net=dict(type='TJ4D_OAS'),
    pts_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=15,
        feat_channels=[64],
        with_distance=False,
        voxel_size=[voxel_size[0] / 2, voxel_size[1] / 2, voxel_size[2]],
        point_cloud_range=point_cloud_range),
    pts_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[496, 432]),
    pts_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[128, 128, 128]),
    projection_voxel_layer=dict(
        max_num_points=12,
        point_cloud_range=projection_cloud_range,
        voxel_size=projection_voxel_size,
        max_voxels=(16000, 40000)
    ),

    projection_voxel_encoder=dict(
        type='PillarFeatureNet',
        in_channels=15,
        feat_channels=[64],
        with_distance=False,
        voxel_size=projection_voxel_size,
        point_cloud_range=projection_cloud_range
    ),
    projection_middle_encoder=dict(
        type='PointPillarsScatter', in_channels=64, output_shape=[144, 256]),
    projection_backbone=dict(
        type='SECOND',
        in_channels=64,
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        out_channels=[64, 128, 256]),
    projection_neck=dict(
        type='SECONDFPN',
        in_channels=[64, 128, 256],
        upsample_strides=[1, 2, 4],
        out_channels=[64, 64, 64]),
    fusion_layer=dict(
      type='DFA',
      img_channels=256,
      rad_channels=384,
    ),
    bbox_head=dict(
        type='Anchor3DHead',
        num_classes=4,
        in_channels=_dim_,
        feat_channels=_dim_,
        use_direction_classifier=True,
        assign_per_class=True,
        anchor_generator=dict(
            type='AlignedAnchor3DRangeGenerator',
            ranges=[
                [0, -40.0, -1.163, 70.4, 40.0, -1.163],
                [0, -40.0, -1.353, 70.4, 40.0, -1.353],
                [0, -40.0, -1.363, 70.4, 40.0, -1.363],
                [0, -40.0, -1.403, 70.4, 40.0, -1.403],
            ],
            sizes=[[0.8, 0.6, 1.69], [1.77, 0.78, 1.60], [4.56, 1.84, 1.70], [10.76, 2.66, 3.47]],
            rotations=[0, 1.57],
            reshape_out=False),
        diff_rad_by_sin=True,
        bbox_coder=dict(type='DeltaXYZWLHRBBoxCoder'),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.SmoothL1Loss', beta=1.0 / 9.0, loss_weight=2.0),
        loss_dir=dict(
            type='mmdet.CrossEntropyLoss', use_sigmoid=False,
            loss_weight=0.2)),
    # model training and testing settings
    train_cfg=dict(
        assigner=[
            dict(  # for Pedestrian
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Cyclist
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.35,
                neg_iou_thr=0.2,
                min_pos_iou=0.2,
                ignore_iof_thr=-1),
            dict(  # for Car
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='mmdet3d.BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,  # 改
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
            dict(  # for Truck
                type='Max3DIoUAssigner',
                iou_calculator=dict(type='BboxOverlapsNearest3D'),
                pos_iou_thr=0.5,  # 改
                neg_iou_thr=0.35,
                min_pos_iou=0.35,
                ignore_iof_thr=-1),
        ],
        allowed_border=0,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        use_rotate_nms=True,
        nms_across_levels=False,
        nms_thr=0.01,
        score_thr=0.1,
        min_bbox_size=0,
        nms_pre=100,
        max_num=50)
)
# dataset settings
dataset_type = 'TJ4DRadSetDataset'
data_root = 'data/TJ4DRadSet'
class_names = ['Pedestrian', 'Cyclist', 'Car', 'Truck']
point_cloud_range = [0.0, -39.68, -3, 69.12, 39.68, 2.76]
input_modality = dict(use_lidar=True, use_camera=True)
metainfo = dict(classes=class_names)
data_prefix=dict(pts='TJ4DRadSet_4DRadar/training/velodyne', img='TJ4DRadSet_Non_Public/training/image_2')
backend_args = None

train_pipeline = [
    dict(
        type='LoadVirtualPointsFromTJ4D',
        coord_type='LIDAR',
        modality='radar',
        load_dim=15,
        use_dim=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        virtual_points_root_path=virtual_points_root_path,
        backend_args=None),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_bbox=True, with_label=True),
    dict(type='ImageAug3D', data_aug_conf=ida_aug_conf, is_train=True),
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CreateProjectionPoints', coord_type='LIDAR', filter_min=0.0, filter_max=80.0),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'projection_points', 'bda_rot', 'cam_aware', 'cam2img', 'radar_depth', 'ori_cam2img', 'lidar2cam',
            'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

test_pipeline = [
    dict(
        type='LoadVirtualPointsFromTJ4D',
        coord_type='LIDAR',
        modality='radar',
        load_dim=15,
        use_dim=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        virtual_points_root_path=virtual_points_root_path,
        backend_args=None),
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='GlobalRotScaleTransFlipAll', bda_aug_conf=bda_aug_conf, is_train=False),
    dict(type='ImageAug3D', data_aug_conf=ida_aug_conf, is_train=False),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='CreateProjectionPoints', coord_type='LIDAR', filter_min=0.0, filter_max=80.0),
    dict(
        type='Pack3DDetInputs',
        keys=[
            'points', 'img', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_bboxes',
            'gt_labels'
        ],
        meta_keys=[
            'projection_points', 'bda_rot', 'cam_aware', 'cam2img', 'radar_depth', 'ori_cam2img', 'lidar2cam',
            'lidar2img', 'cam2lidar',
            'ori_lidar2img', 'img_aug_matrix', 'box_type_3d', 'sample_idx',
            'lidar_path', 'img_path', 'transformation_3d_flow', 'pcd_rotation',
            'pcd_scale_factor', 'pcd_trans', 'img_aug_matrix',
            'lidar_aug_matrix', 'num_pts_feats'
        ])
]

train_dataloader = dict(
    batch_size=3,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file='TJ4DRadSet_infos_train.pkl',
            data_prefix=dict(pts='TJ4DRadSet_4DRadar/training/velodyne', img='TJ4DRadSet_Non_Public/training/image_2'),
            test_mode=False,
            pipeline=train_pipeline,
            modality=input_modality,
            box_type_3d='LiDAR',
            metainfo=metainfo,
            backend_args=backend_args)))
val_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='TJ4DRadSet_infos_valid.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=data_prefix,
        ann_file='TJ4DRadSet_infos_valid.pkl',
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        metainfo=metainfo,
        box_type_3d='LiDAR',
        backend_args=backend_args))
val_evaluator = dict(
    type='TJ4DRadSetMetric',
    ann_file=data_root + '/TJ4DRadSet_infos_valid.pkl',
    metric='bbox',
    backend_args=backend_args)
test_evaluator = val_evaluator

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1, max_keep_ckpts=5),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='Det3DVisualizationHook'))

lr = 1e-4
# This schedule is mainly used by models on nuScenes dataset
# max_norm=10 is better for SECOND
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2))
# learning rate
param_scheduler = [
    # learning rate scheduler
    # During the first 8 epochs, learning rate increases from 0 to lr * 10
    # during the next 12 epochs, learning rate decreases from lr * 10 to
    # lr * 1e-4
    dict(
        type='CosineAnnealingLR',
        T_max=4,
        eta_min=lr * 8,
        begin=0,
        end=4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingLR',
        T_max=8,
        eta_min=lr * 1e-4,
        begin=4,
        end=12,
        by_epoch=True,
        convert_to_iter_based=True),
    # momentum scheduler
    # During the first 8 epochs, momentum increases from 0 to 0.85 / 0.95
    # during the next 12 epochs, momentum increases from 0.85 / 0.95 to 1
    dict(
        type='CosineAnnealingMomentum',
        T_max=4,
        eta_min=0.85 / 0.95,
        begin=0,
        end=4,
        by_epoch=True,
        convert_to_iter_based=True),
    dict(
        type='CosineAnnealingMomentum',
        T_max=8,
        eta_min=1,
        begin=4,
        end=12,
        by_epoch=True,
        convert_to_iter_based=True)
]

# runtime settings
train_cfg = dict(by_epoch=True, max_epochs=12, val_interval=1)
val_cfg = dict()
test_cfg = dict()

auto_scale_lr = dict(enable=False, base_batch_size=12)

load_from = '/home/chenjh/Documents/mmdetection3d/ckpts/img-checkpoint.pth'
