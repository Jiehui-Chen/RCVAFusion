# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py
import argparse
import multiprocessing as mp
import os

# fmt: off
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# fmt: on
import torch
import time

import numpy as np

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
from mask2former import add_maskformer2_config
from predictor import VisualizationDemo
from utils import get_fov_mask, read_calib


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_maskformer2_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument("--video-input", help="Path to video file.")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )
    parser.add_argument(
        "--pts-save-path",
        required=True,
        help="path to save path of hybrid points",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser


def read_file(path):
    points = np.fromfile(path, dtype=np.float32).reshape(-1, 7)
    return points

def process_one_frame(instances,points,extrinsic,intrinsic):
    labels = instances.pred_classes
    scores = instances.scores
    masks = instances.pred_masks  # 类别数量 * pixel数量

    # remove empty mask and their scores / labels
    empty_mask = masks.reshape(scores.shape[0], W * H).sum(dim=1) == 0

    labels = labels[~empty_mask]
    scores = scores[~empty_mask]
    masks = masks[~empty_mask]

    score_mask = scores > 0.1

    labels = labels[score_mask]
    scores = scores[score_mask]
    masks = masks[score_mask]


    points_in_img, mask = get_fov_mask(points[:, :3], extrinsic, intrinsic, h=1216, w=1936)

    pts_img = points_in_img[:, :2]
    pts_rect_depth = points_in_img[:, 2]
    points_in = points[mask]
    other_points = points[~mask]

    #points depth filter
    depth_mask = pts_rect_depth < 51.2
    pts_rect_depth = pts_rect_depth[depth_mask]
    points_in = points_in[depth_mask]
    pts_img = pts_img[depth_mask]

    one_hot_labels = []
    for i in range(8):
        one_hot_label = torch.zeros(8, device='cuda:0', dtype=torch.float32)
        one_hot_label[i] = 1
        one_hot_labels.append(one_hot_label)

    one_hot_labels = torch.stack(one_hot_labels, dim=0)
    transformed_labels = one_hot_labels.gather(0, labels.reshape(-1, 1).repeat(1, 8))
    # transformed_labels = torch.cat([transformed_labels, scores.unsqueeze(-1)], dim=1)  # 转成one hot形式
    res = add_virtual_mask(masks,transformed_labels,points_in,pts_img,pts_rect_depth,extrinsic,intrinsic,other_points)
    return res

def to_tensor(arr, dtype):
    return torch.tensor(arr, dtype=dtype, device='cuda:0')

def is_within_mask(points_xyc, masks, H=900, W=1600):
    points_xyc = points_xyc.long()
    valid = masks[:, points_xyc[:, 0], points_xyc[:, 1]]
    return valid.transpose(1, 0)

def add_virtual_mask(seg_masks, seg_labels, points, pts_img, pts_rect_depth,extrinsic, intrinsic, other_points,DIST_THRESH = 3000):
    NUM_POINT = 500
    masks = to_tensor(seg_masks, dtype=torch.bool).permute(0, 2, 1)  # 370 * 1224 to 1224 x 370
    labels = to_tensor(seg_labels, dtype=torch.float32)
    pts_img = to_tensor(pts_img, dtype=torch.float32)
    pts_rect_depth = to_tensor(pts_rect_depth, dtype=torch.float32)

    valid = is_within_mask(pts_img, masks)

    foreground_real_point_mask = valid.sum(dim=1) > 0

    if valid.shape[1] == 0:
        other_points = np.concatenate([points, other_points], axis=0)
        result_obj = {
            'virtual_points': None,
            'real_points': None,
            'other_points': other_points
        }
        print('non_valid')
        return result_obj

    foreground_real_point_instance_ids = (valid.float().argmax(dim=1)+1)[foreground_real_point_mask]
    foreground_real_point_labels = labels[foreground_real_point_instance_ids-1]
    foreground_real_point_feature = to_tensor(points[:, 3:], dtype=torch.float32)[foreground_real_point_mask]

    foreground_real_points = torch.cat(
        [pts_img[foreground_real_point_mask], foreground_real_point_instance_ids.unsqueeze(-1)],
        dim=-1
    )

    if len(foreground_real_points) == 0:
        other_points = np.concatenate([points, other_points], axis=0)
        result_obj = {
            'virtual_points': None,
            'real_points': None,
            'other_points': other_points
        }
        print('non_foreground_points')
        return result_obj

    offsets = []
    for idx, mask in enumerate(masks):
        indices = mask.nonzero()
        selected_indices = torch.randperm(len(indices))[:NUM_POINT].to(masks.device)  # 从mask中随机挑取采样点
        if len(selected_indices) < NUM_POINT:
            selected_indices = torch.cat([selected_indices, selected_indices[
                selected_indices.new_zeros(NUM_POINT - len(selected_indices))]])

        offset = indices[selected_indices]
        offsets.append(offset)


    offsets = torch.stack(offsets, dim=0)
    virtual_point_instance_ids = torch.arange(1, 1 + masks.shape[0],
                                              dtype=torch.float32, device='cuda:0').reshape(masks.shape[0], 1,
                                                                                            1).repeat(1, NUM_POINT, 1)
    virtual_points_xyi = torch.cat([offsets, virtual_point_instance_ids], dim=-1).reshape(-1, 3)

    # avoid matching acroos instances
    foreground_real_points[:, -1] *= 1e5
    virtual_points_xyi[:, -1] *= 1e5

    all_virtual_points = []

    cur_virtual_points = virtual_points_xyi
    dist = torch.norm(cur_virtual_points.unsqueeze(1) - foreground_real_points.unsqueeze(0), dim=-1)  # N_virtual * N_real
    k_min = min(1, foreground_real_points.shape[0])
    k_nearest_dist, k_nearest_indices = torch.topk(dist, k=k_min, dim=1, largest=False)  # N_virtual
    for j in range(k_min):
        nearest_dist = k_nearest_dist[:, j]
        local_nearest_indices = k_nearest_indices[:, j]
        mask = nearest_dist < DIST_THRESH

        foreground_indices = foreground_real_point_mask.nonzero().reshape(-1)

        local_nearest_indices = local_nearest_indices[mask]
        global_nearest_indices = foreground_indices[local_nearest_indices]
        virtual_points = cur_virtual_points[mask]

        # finish matching, infer depth
        virtual_point_depth = pts_rect_depth[global_nearest_indices]
        virtual_point_label = foreground_real_point_labels[local_nearest_indices]
        virtual_point_feature = foreground_real_point_feature[local_nearest_indices]

        # back project virtual points into lidar coordinate
        virtual_points = virtual_points[:, :2].cpu().numpy()
        virtual_point_depth = virtual_point_depth.cpu().numpy()
        virtual_point_label = virtual_point_label.cpu().numpy()
        virtual_point_feature = virtual_point_feature.cpu().numpy()

        virtual_points = virtual_points * virtual_point_depth.reshape(-1, 1)
        virtual_points = np.concatenate([virtual_points, virtual_point_depth.reshape(-1, 1)], axis=-1)
        point_in_camera = np.matmul(np.linalg.inv(intrinsic[:, :3]), virtual_points.T)
        point_in_camera = point_in_camera.T
        point_in_camera = np.hstack((point_in_camera, np.ones(shape=(point_in_camera.shape[0], 1)))).T
        virtual_points_lidar = np.matmul(np.linalg.inv(extrinsic), point_in_camera)
        virtual_points_lidar = virtual_points_lidar.T[:, 0:3]

        virtual_points = np.concatenate([
            virtual_points_lidar[:, :3], virtual_point_feature, virtual_point_label
        ], axis=1)
        all_virtual_points.append(virtual_points)

    virtual_points = np.concatenate(all_virtual_points, axis=0)

    # get foreground real points
    foreground_real_points = points[foreground_real_point_mask.cpu().numpy()]
    non_foreground_real_points = points[~foreground_real_point_mask.cpu().numpy()]
    foreground_real_point_labels = foreground_real_point_labels.cpu().numpy()
    foreground_real_point_feature = foreground_real_point_feature.cpu().numpy()

    real_points = np.concatenate([
        foreground_real_points[:, :3], foreground_real_point_feature,
        foreground_real_point_labels
    ], axis=1)

    other_points = np.concatenate([non_foreground_real_points, other_points],axis=0)

    result_obj = {
        'virtual_points': virtual_points,
        'real_points': real_points,
        'other_points': other_points
    }
    return result_obj

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)

    demo = VisualizationDemo(cfg)
    txt_path = 'data/VoD/lidar/ImageSets/train_val.txt'
    path_base = 'data/VoD'
    radar_base = 'radar_5frames'
    save_path = args.pts_save_path
    with open(txt_path, 'r') as file:
        prefixes = file.read().split()
    W=1936
    H=1216
    for prefix in prefixes:

        npy_filename = f"{prefix}.npy"
        radar_path = os.path.join(path_base, radar_base, 'training/velodyne', prefix + '.bin')
        calib_path = os.path.join(path_base, radar_base, 'training/calib', prefix + '.txt')
        img_path = os.path.join(path_base, radar_base,'training/image_2',prefix + '.jpg')
        img = read_image(img_path, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)

        instances = predictions['instances']

        _, _, intrinsic, _, R0, extrinsic = read_calib(calib_path)

        radar_points = read_file(radar_path)

        res = process_one_frame(instances,radar_points,extrinsic,intrinsic)

        np.save(os.path.join(save_path,npy_filename), res)
        print(npy_filename)
