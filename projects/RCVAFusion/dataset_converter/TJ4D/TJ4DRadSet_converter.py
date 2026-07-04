import os
from pathlib import Path
import mmengine
import numpy as np
from mmdet3d.structures.ops import box_np_ops
from projects.RCVAFusion.dataset_converter.TJ4D.TJ4DRadSet_data_utlis import get_TJ4DRadSet_image_info


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def create_TJ4DRadSet_info_file(root_path='data/TJ4DRadSet'):
    imageset_folder = os.path.join(root_path, 'TJ4DRadSet_4DRadar','ImageSets')
    train_ids = _read_imageset_file(os.path.join(imageset_folder, 'train.txt'))
    valid_ids = _read_imageset_file(os.path.join(imageset_folder, 'val.txt'))
    test_ids = _read_imageset_file(os.path.join(imageset_folder, 'test.txt'))

    print('Generate info.pkl this may take several minutes.')

    info_train_path = os.path.join(root_path, 'infos_train.pkl')
    if not os.path.exists(info_train_path):
        print(f'{info_train_path} start to create.')
        infos_train = get_TJ4DRadSet_image_info(
            path=root_path,
            velodyne=True,
            calib=True,
            image_ids=train_ids,
            relative_path=True)
        _calculate_num_points_in_gt(root_path, infos_train, relative_path=True,remove_outside=False)
        mmengine.dump(infos_train, info_train_path)
        print(f'{info_train_path} create successfully.')
    else:
        print(f'{info_train_path} already exists, skip.')

    info_valid_path = os.path.join(root_path, 'infos_valid.pkl')
    if not os.path.exists(info_valid_path):
        print(f'{info_valid_path} start to create.')
        infos_valid = get_TJ4DRadSet_image_info(
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
        infos_test = get_TJ4DRadSet_image_info(
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
                                num_features=8):
    for info in mmengine.track_iter_progress(infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape((-1, num_features))
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info['image_shape'])

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
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        # num_ignored = len(annos['dimensions']) - num_obj
        # num_points_in_gt = np.concatenate(
        #     [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)

def create_TJ4DRadSet_conditions_info_file(root_path='data/TJ4DRadSet'):
    # imageset_folder = os.path.join(root_path, 'TJ4DRadSet_4DRadar','ImageSets')
    dark_ids = _read_imageset_file(os.path.join(root_path, 'val_dark.txt'))
    normal_ids = _read_imageset_file(os.path.join(root_path, 'val_normal.txt'))
    shiny_ids = _read_imageset_file(os.path.join(root_path, 'val_shiny.txt'))

    print('Generate info.pkl this may take several minutes.')

    info_dark_path = os.path.join(root_path, 'conditions', 'infos_dark.pkl')
    if not os.path.exists(info_dark_path):
        print(f'{info_dark_path} start to create.')
        infos_train = get_TJ4DRadSet_image_info(
            path=root_path,
            velodyne=True,
            calib=True,
            image_ids=dark_ids,
            relative_path=True)
        _calculate_num_points_in_gt(root_path, infos_train, relative_path=True,remove_outside=False)
        mmengine.dump(infos_train, info_dark_path)
        print(f'{info_dark_path} create successfully.')
    else:
        print(f'{info_dark_path} already exists, skip.')

    info_normal_path = os.path.join(root_path, 'conditions', 'infos_normal.pkl')
    if not os.path.exists(info_normal_path):
        print(f'{info_normal_path} start to create.')
        infos_valid = get_TJ4DRadSet_image_info(
            path=root_path,
            velodyne=True,
            calib=True,
            image_ids=normal_ids,
            relative_path=True
        )
        _calculate_num_points_in_gt(root_path, infos_valid, relative_path=True, remove_outside=False)
        mmengine.dump(infos_valid, info_normal_path)
        print(f'{info_normal_path} create successfully.')
    else:
        print(f'{info_normal_path} already exists, skip.')

    info_shiny_path = os.path.join(root_path, 'conditions', 'infos_shiny.pkl')
    if not os.path.exists(info_shiny_path):
        print(f'{info_shiny_path} start to create.')
        infos_valid = get_TJ4DRadSet_image_info(
            path=root_path,
            velodyne=True,
            calib=True,
            image_ids=shiny_ids,
            relative_path=True
        )
        _calculate_num_points_in_gt(root_path, infos_valid, relative_path=True, remove_outside=False)
        mmengine.dump(infos_valid, info_shiny_path)
        print(f'{info_shiny_path} create successfully.')
    else:
        print(f'{info_shiny_path} already exists, skip.')
