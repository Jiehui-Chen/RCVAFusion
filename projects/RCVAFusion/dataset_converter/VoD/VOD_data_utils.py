import os
from collections import OrderedDict
from concurrent import futures as futures
from pathlib import Path

import mmengine
import numpy as np
from PIL import Image
from skimage import io

def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:05d}'.format(img_idx)

def get_velodyne_path(idx,
                      prefix,
                      relative_path=True,
                      use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += '.bin'
    base_lidar_velodyne='lidar/training/velodyne'
    base_radar_velodyne='radar_5frames/training/velodyne'
    path_lidar=os.path.join(base_lidar_velodyne,img_idx_str)
    path_radar = os.path.join(base_radar_velodyne, img_idx_str)
    if relative_path:
        return path_lidar,path_radar
    else:
        return os.path.join(prefix,path_lidar),os.path.join(prefix,path_radar)

def get_image_path(idx,
                   prefix,
                   relative_path=True,
                   use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += '.jpg'
    base_img = 'lidar/training/image_2'
    path = os.path.join(base_img, img_idx_str)
    if relative_path:
        return path
    else:
        return os.path.join(prefix, path)
def get_label_path(idx,
                   prefix,
                   relative_path=True,
                   use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += '.txt'
    base_label = 'lidar/training/label_2'
    path = os.path.join(base_label, img_idx_str)
    if relative_path:
        return path
    else:
        return os.path.join(prefix, path)

def get_calib_path(idx,
                   prefix,
                   relative_path=True,
                   use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += '.txt'
    base_lidar_calib='lidar/training/calib'
    base_radar_calib='radar_5frames/training/calib'
    path_lidar=os.path.join(base_lidar_calib,img_idx_str)
    path_radar = os.path.join(base_radar_calib, img_idx_str)
    if relative_path:
        return path_lidar,path_radar
    else:
        return os.path.join(prefix,path_lidar),os.path.join(prefix,path_radar)

def get_calib(calib_path,extend_matrix=True):
    calib_info = {}
    with open(calib_path, 'r') as f:
        lines = f.readlines()
    P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                   ]).reshape([3, 4])
    P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                   ]).reshape([3, 4])
    P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                   ]).reshape([3, 4])
    P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                   ]).reshape([3, 4])
    if extend_matrix:
        P0 = _extend_matrix(P0)
        P1 = _extend_matrix(P1)
        P2 = _extend_matrix(P2)
        P3 = _extend_matrix(P3)
    R0_rect = np.array([
        float(info) for info in lines[4].split(' ')[1:10]
    ]).reshape([3, 3])
    if extend_matrix:
        rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
        rect_4x4[3, 3] = 1.
        rect_4x4[:3, :3] = R0_rect
    else:
        rect_4x4 = R0_rect

    Tr_velo_to_cam = np.array([
        float(info) for info in lines[5].split(' ')[1:13]
    ]).reshape([3, 4])
    # Tr_imu_to_velo = np.array([
    #     float(info) for info in lines[6].split(' ')[1:13]
    # ]).reshape([3, 4])
    if extend_matrix:
        Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
        # Tr_imu_to_velo = _extend_matrix(Tr_imu_to_velo)
    calib_info['P0'] = P0
    calib_info['P1'] = P1
    calib_info['P2'] = P2
    calib_info['P3'] = P3
    calib_info['R0_rect'] = rect_4x4
    calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam
    # calib_info['Tr_imu_to_velo'] = Tr_imu_to_velo
    return calib_info

def get_VOD_image_info(path,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         image_ids=7481,
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True):
    """
    KITTI annotation format version 2:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam: ...
            P2: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        radar_pc_info = {'num_features': 7}
        lidar_pc_info = {'num_features': 4}
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            lidar_pc_info['velodyne_path'],radar_pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, relative_path)
        image_info['image_path'] = get_image_path(idx, path,
                                                  relative_path)
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path)
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)
        if label_info:
            label_path = get_label_path(idx, path, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['lidar_point_cloud'] = lidar_pc_info
        info['radar_point_cloud'] = radar_pc_info
        if calib:
            lidar_calib_path,radar_calib_path = get_calib_path(
                idx, path, relative_path=False)
            calib_lidar=get_calib(lidar_calib_path)
            calib_radar=get_calib(radar_calib_path)
            calib_info['calib_lidar']=calib_lidar
            calib_info['calib_radar']=calib_radar
            info['calib'] = calib_info
        if annotations is not None:
            info['annos'] = annotations
            # add_difficulty_to_annos(info)
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)




def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat