import numpy as np
import torch

def get_fov_mask(point_in_lidar, extrinsic, intrinsic, h, w):
    """
    获取fov内的点云mask, 即能够投影在图像上的点云mask
    :param point_in_lidar:   雷达点云 numpy.ndarray `N x 3`
    :param extrinsic:        外参 numpy.ndarray `4 x 4`
    :param intrinsic:        内参 numpy.ndarray `3 x 3` or `3 x 4`
    :param h:                图像高 int
    :param w:                图像宽 int
    :return: point_in_image: (u, v, z)  numpy.ndarray `N x 3`
    :return:                 numpy.ndarray  `1 x N`
    """
    point_in_image = lidar2image(point_in_lidar, extrinsic, intrinsic)
    front_bound = point_in_image[:, -1] > 0
    point_in_image[:, 0] = np.round(point_in_image[:, 0])
    point_in_image[:, 1] = np.round(point_in_image[:, 1])
    u_bound = np.logical_and(point_in_image[:, 0] >= 0, point_in_image[:, 0] < w)
    v_bound = np.logical_and(point_in_image[:, 1] >= 0, point_in_image[:, 1] < h)
    uv_bound = np.logical_and(u_bound, v_bound)
    mask = np.logical_and(front_bound, uv_bound)
    return point_in_image[mask], mask

def lidar2image(point_in_lidar, extrinsic, intrinsic):
    """
    雷达系到图像系投影  获得(u, v, z)
    :param point_in_lidar: numpy.ndarray `N x 3`
    :param extrinsic: numpy.ndarray `4 x 4`
    :param intrinsic: numpy.ndarray `3 x 3` or `3 x 4`
    :return: point_in_image numpy.ndarray `N x 3` (u, v, z)
    """
    point_in_camera = lidar2camera(point_in_lidar, extrinsic)
    point_in_image = camera2image(point_in_camera, intrinsic)
    return point_in_image
def lidar2camera(point_in_lidar, extrinsic):
    """
    雷达系到相机系投影
    :param point_in_lidar: numpy.ndarray `N x 3`
    :param extrinsic: numpy.ndarray `4 x 4`
    :return: point_in_camera numpy.ndarray `N x 3`
    """
    point_in_lidar = np.hstack((point_in_lidar, np.ones(shape=(point_in_lidar.shape[0], 1)))).T
    point_in_camera = np.matmul(extrinsic, point_in_lidar)[:-1, :]  # (X, Y, Z)
    point_in_camera = point_in_camera.T
    return point_in_camera
def camera2image(point_in_camera, intrinsic):
    """
    相机系到图像系投影
    :param point_in_camera: point_in_camera numpy.ndarray `N x 3`
    :param intrinsic: numpy.ndarray `3 x 3` or `3 x 4`
    :return: point_in_image numpy.ndarray `N x 3` (u, v, z)
    """
    point_in_camera = point_in_camera.T
    point_z = point_in_camera[-1]

    if intrinsic.shape == (3, 3):  # 兼容kitti的P2, 对于没有平移的intrinsic添0
        intrinsic = np.hstack((intrinsic, np.zeros((3, 1))))

    point_in_camera = np.vstack((point_in_camera, np.ones((1, point_in_camera.shape[1]))))
    point_in_image = (np.matmul(intrinsic, point_in_camera) / point_z)  # 向图像上投影
    point_in_image[-1] = point_z
    point_in_image = point_in_image.T
    return point_in_image


def read_calib(calib_path):

    with open(calib_path, 'r') as f:
        raw = f.readlines()
    P0 = np.array(list(map(float, raw[0].split()[1:]))).reshape((3, 4))
    P1 = np.array(list(map(float, raw[1].split()[1:]))).reshape((3, 4))
    P2 = np.array(list(map(float, raw[2].split()[1:]))).reshape((3, 4))
    P3 = np.array(list(map(float, raw[3].split()[1:]))).reshape((3, 4))
    R0 = np.array(list(map(float, raw[4].split()[1:]))).reshape((3, 3))
    R0 = np.hstack((R0, np.array([[0], [0], [0]])))
    R0 = np.vstack((R0, np.array([0, 0, 0, 1])))
    lidar2camera_m = np.array(list(map(float, raw[5].split()[1:]))).reshape((3, 4))
    lidar2camera_m = np.vstack((lidar2camera_m, np.array([0, 0, 0, 1])))

    return P0, P1, P2, P3, R0, lidar2camera_m

def get_label_length(x):
    if x ==0:
        return 1.6
    elif x==1:
        return 2.0
    elif 2 <= x <= 5:
        return 5.0
    elif x in (6, 7):
        return 2.0
    else:
        return None  # 或抛出异常 raise ValueError("输入值未定义")

def forground_filter(valid,pts_rect_depth,foreground_real_point_mask, labels_without_encoder):
    N_points, N_valid = valid.shape
    assert N_valid == len(labels_without_encoder)
    for i in range(N_valid):
        label = labels_without_encoder[i]
        depth_length = get_label_length(label)
        cur_valid = valid[:,i]
        cur_mask_object = cur_valid == True
        if cur_mask_object.sum() > 0:

            indices = torch.where(cur_mask_object)[0]
            cur_object_depth = pts_rect_depth[cur_mask_object]

            min_depth = torch.min(cur_object_depth)
            threshold = min_depth + depth_length
            cur_mask_depth = cur_object_depth > threshold

            exclude_indices = indices[cur_mask_depth]
            for j in range(len(exclude_indices)):
                if foreground_real_point_mask[exclude_indices[j]] == True:
                    foreground_real_point_mask[exclude_indices[j]] = False

    return foreground_real_point_mask

def get_cur_mask_depth(depth_data,depth_threshold):
    # Step 1: 创建候选mask（深度大于阈值的区域）
    mask_candidate = depth_data > depth_threshold
    if mask_candidate.sum()==0:
        return mask_candidate

    if mask_candidate.sum() < 0.5 * len(mask_candidate):
        return mask_candidate

    # Step 2: 获取所有满足条件点的索引和深度值
    indices = np.where(mask_candidate)
    depths_valid = depth_data[mask_candidate]


    # Step 4: 按分数降序排列并选择前80%
    sorted_indices = np.argsort(depths_valid)  # 降序排列的索引
    k = int(round(0.7 * len(depths_valid)))  # 计算需要选择的点数

    # Step 5: 创建最终mask
    selected_mask = np.zeros_like(mask_candidate, dtype=bool)
    selected_indices = indices[0][sorted_indices[k:]]  # 对多维索引的支持
    selected_mask[selected_indices] = True
    return selected_mask

def get_cur_mask_depth_2(depth_data,depth_threshold):
    ratio=0.3
    # Step 1: 创建候选mask（深度大于阈值的区域）
    mask_candidate = depth_data > depth_threshold
    if mask_candidate.sum()==0:
        return mask_candidate

    if mask_candidate.sum() < (1-ratio) * len(mask_candidate):
        return mask_candidate

    indices = np.where(mask_candidate)
    depths_valid = depth_data[mask_candidate]

    sorted_indices = np.argsort(depths_valid)  #lift排列的索引
    k = int(round((1-ratio) * len(mask_candidate)))  # 计算需要选择的点数
    k = max(1,k)

    # Step 5: 创建最终mask
    selected_mask = np.zeros_like(mask_candidate, dtype=bool)
    selected_indices = indices[0][sorted_indices[-k:]]  # 对多维索引的支持
    selected_mask[selected_indices] = True
    return selected_mask

def to_tensor(arr, dtype):
    return torch.tensor(arr, dtype=dtype, device='cuda:0')

def forground_filter_one(valid,pts_rect_depth,foreground_real_point_mask, labels_without_encoder):
    N_points, N_valid = valid.shape
    assert N_valid == len(labels_without_encoder)
    for i in range(N_valid):
        label = labels_without_encoder[i]
        depth_length = get_label_length(label)
        cur_valid = valid[:,i]
        cur_mask_object = cur_valid == True
        if cur_mask_object.sum() > 0:

            indices = torch.where(cur_mask_object)[0]
            cur_object_depth = pts_rect_depth[cur_mask_object]

            min_depth = torch.min(cur_object_depth)
            depth_threshold = min_depth + depth_length
            cur_mask_depth = get_cur_mask_depth_2(cur_object_depth.to('cpu').numpy(), depth_threshold.to('cpu').numpy())
            cur_mask_depth = to_tensor(cur_mask_depth, dtype=torch.bool)
            exclude_indices = indices[cur_mask_depth]
            for j in range(len(exclude_indices)):
                if foreground_real_point_mask[exclude_indices[j]] == True:
                    foreground_real_point_mask[exclude_indices[j]] = False

    return foreground_real_point_mask