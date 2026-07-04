import os
from pathlib import Path
import mmengine
import numpy as np
from mmdet3d.structures.ops import box_np_ops
from projects.RCVAFusion.dataset_converter.VoD.VOD_data_utils import get_VOD_image_info


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def create_VOD_info_file(root_path='data/VoD'):
    imageset_folder = os.path.join(root_path,'lidar','ImageSets')
    train_ids = _read_imageset_file(os.path.join(imageset_folder, 'train.txt'))
    valid_ids = _read_imageset_file(os.path.join(imageset_folder, 'val.txt'))
    test_ids = _read_imageset_file(os.path.join(imageset_folder, 'test.txt'))
    print('Generate info.pkl this may take several minutes.')

    info_train_path = os.path.join(root_path, 'infos_train.pkl')
    if not os.path.exists(info_train_path):
        print(f'{info_train_path} start to create.')
        infos_train = get_VOD_image_info(
            path=root_path,
            velodyne=True,
            calib=True,
            image_ids=train_ids,
            relative_path=True)
        _calculate_num_points_in_gt(root_path, infos_train, relative_path=True, remove_outside=False)
        mmengine.dump(infos_train, info_train_path)
        print(f'{info_train_path} create successfully.')
    else:
        print(f'{info_train_path} already exists, skip.')

    info_valid_path = os.path.join(root_path, 'infos_valid.pkl')
    if not os.path.exists(info_valid_path):
        print(f'{info_valid_path} start to create.')
        infos_valid = get_VOD_image_info(
            path=root_path,
            velodyne=True,
            calib=True,
            image_ids=valid_ids,
            relative_path=True
        )
        _calculate_num_points_in_gt(root_path, infos_valid, relative_path=True, remove_outside=False)
        mmengine.dump(infos_valid, info_valid_path)
        print(f'{info_valid_path} create successfully.')
    else:
        print(f'{info_valid_path} already exists, skip.')

    info_trainval_path = os.path.join(root_path, 'infos_trainval.pkl')
    if not os.path.exists(info_trainval_path):
        print(f'{info_trainval_path} start to create.')
        infos_train = mmengine.load(info_train_path)
        infos_valid = mmengine.load(info_valid_path)
        mmengine.dump(infos_train + infos_valid, info_trainval_path)
        print(f'{info_trainval_path} create successfully.')
    else:
        print(f'{info_trainval_path} already exists, skip.')

    info_test_path = os.path.join(root_path, 'infos_test.pkl')
    if not os.path.exists(info_test_path):
        print(f'{info_test_path} start to create.')
        infos_test = get_VOD_image_info(
            path=root_path,
            label_info=False,
            velodyne=True,
            calib=True,
            image_ids=test_ids,
            relative_path=True
        )
        mmengine.dump(infos_test, info_test_path)
        print(f'{info_test_path} create successfully.')
    else:
        print(f'{info_test_path} already exists, skip.')


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features_lidar=4,
                                num_features_radar=7):
    for info in mmengine.track_iter_progress(infos):
        lidar_pc_info = info['lidar_point_cloud']
        radar_pc_info = info['radar_point_cloud']
        image_info = info['image']
        calib = info['calib']
        if relative_path:
            lidar_v_path = str(Path(data_path) / lidar_pc_info['velodyne_path'])
            radar_v_path = str(Path(data_path) / radar_pc_info['velodyne_path'])
        else:
            lidar_v_path = lidar_pc_info['velodyne_path']
            radar_v_path = radar_pc_info['velodyne_path']
        lidar_points_v = np.fromfile(
            lidar_v_path, dtype=np.float32, count=-1).reshape((-1, num_features_lidar))
        radar_points_v = np.fromfile(
            radar_v_path, dtype=np.float32, count=-1).reshape((-1, num_features_radar))
        rect_lidar = calib['calib_lidar']['R0_rect']
        rect_radar=calib['calib_radar']['R0_rect']
        Trv2c_lidar = calib['calib_lidar']['Tr_velo_to_cam']
        Trv2c_radar=calib['calib_radar']['Tr_velo_to_cam']
        P2_lidar = calib['calib_lidar']['P2']
        P2_radar = calib['calib_radar']['P2']
        if remove_outside:
            lidar_points_v = box_np_ops.remove_outside_points(
                lidar_points_v, rect_lidar, Trv2c_lidar, P2_lidar, image_info['image_shape'])
            radar_points_v = box_np_ops.remove_outside_points(
                radar_points_v, rect_radar, Trv2c_radar, P2_radar, image_info['image_shape'])
        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        # num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions']
        loc = annos['location']
        rots = annos['rotation_y']
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect_lidar, Trv2c_lidar)
        gt_boxes_radar=box_np_ops.box_camera_to_lidar(
            gt_boxes_camera,rect_radar,Trv2c_radar
        )

        indices = box_np_ops.points_in_rbbox(lidar_points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        # num_ignored = len(annos['dimensions']) - num_obj
        # num_points_in_gt = np.concatenate(
        #     [num_points_in_gt, -np.ones([num_ignored])])
        annos['lidar_num_points_in_gt'] = num_points_in_gt.astype(np.int32)

        indices = box_np_ops.points_in_rbbox(radar_points_v[:, :3], gt_boxes_radar)
        num_points_in_gt = indices.sum(0)
        annos['radar_num_points_in_gt'] = num_points_in_gt.astype(np.int32)
