#!/usr/bin/env bash
# Import existing datasets from server to this project using soft link

# nuScenes
if [ ! -d "./data/nuscenes" ]; then
  echo "nuScenes imported"
  ln -s /mnt/ChillDisk/dataset/mmdetection3d/nuscenes/ ./data/nuscenes
else
  echo "nuScenes existed"
fi

# KITTI
if [ ! -d "./data/kitti" ]; then
  echo "KITTI imported"
  ln -s /mnt/ChillDisk/dataset/mmdetection3d/kitti/ ./data/kitti
else
  echo "KITTI existed"
fi
