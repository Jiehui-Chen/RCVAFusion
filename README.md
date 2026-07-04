# RCVAFusion

RCVAFusion is a radar-camera 3D object detection project built on top of
MMDetection3D. This repository contains the model, dataset converters, custom
CUDA operators, and configs for View-of-Delft (VoD) and TJ4DRadSet experiments.

## Environment

The current development environment used for this project is:

| Component | Version |
| --- | --- |
| OS | Ubuntu 20.04.6 LTS |
| Python | 3.8 |
| CUDA | 11.7 |
| PyTorch | 2.0.1 |
| TorchVision | 0.15.2 |
| TorchAudio | 2.0.2 |
| MMEngine | 0.9.0 |
| MMCV | 2.0.1 |
| MMDetection | 3.1.0 |
| MMDetection3D | this repository, editable install |
| spconv | spconv-cu117 |

You can create the environment with the helper script:

```bash
bash setup_environment.sh
```

Build the RCVAFusion custom operators:

```bash
python projects/RCVAFusion/setup.py build_ext --inplace
```

If CUDA is available but not detected by PyTorch during build, force CUDA
compilation:

```bash
FORCE_CUDA=1 python projects/RCVAFusion/setup.py build_ext --inplace
```

The setup builds:

```text
projects.RCVAFusion.mmdet3d_plugin.models.sub_models.ops.bev_pool.bev_pool_ext
projects.RCVAFusion.mmdet3d_plugin.models.sub_models.ops.voxel.voxel_layer
```

## Data Preparation

All dataset paths below are relative to the repository root. The configs expect
the root data directory to be:

```text
data/
```

The converter scripts write final MMDetection3D-style info files directly under
each dataset root. Intermediate files named `infos_train.pkl`,
`infos_valid.pkl`, `infos_trainval.pkl`, and `infos_test.pkl` are removed by
default after conversion.

### View-of-Delft (VoD)

Download and extract the official View-of-Delft public dataset. From the
downloaded `view_of_delft_PUBLIC` directory, place or link the following
subdirectories into `data/VoD`:

```text
data/VoD/
├── lidar/
│   ├── ImageSets/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── train_val.txt
│   │   └── test.txt
│   └── training/
│       ├── calib/
│       ├── image_2/
│       ├── label_2/
│       └── velodyne/
├── radar/
├── radar_3frames/
└── radar_5frames/
    └── training/
        ├── calib/
        ├── image_2/
        └── velodyne/
```

The files can be copied, or symbolic links can be used. For example:

```bash
mkdir -p data/VoD
ln -s /path/to/view_of_delft_PUBLIC/lidar data/VoD/lidar
ln -s /path/to/view_of_delft_PUBLIC/radar data/VoD/radar
ln -s /path/to/view_of_delft_PUBLIC/radar_3frames data/VoD/radar_3frames
ln -s /path/to/view_of_delft_PUBLIC/radar_5frames data/VoD/radar_5frames
```

Generate the VoD info files:

```bash
python projects/RCVAFusion/create_vod.py
```

Expected generated files:

```text
data/VoD/
├── VOD_infos_train.pkl
├── VOD_infos_valid.pkl
├── VOD_infos_trainval.pkl
└── VOD_infos_test.pkl
```

The VoD config uses:

```python
data_root = 'data/VoD'
data_prefix = dict(
    pts_lidar='lidar/training/velodyne',
    img='lidar/training/image_2',
    pts_radar='radar_5frames/training/velodyne')
```

For RCVAFusion training, virtual point files are also expected at:

```text
data/VoD/vod_virtual_points/
```

Each file should be named by sample id, for example:

```text
data/VoD/vod_virtual_points/00001.npy
```

The helper script `virtual_points_demo/vod_visual_points.py` is configured to
read VoD data from `data/VoD` and can be used to generate these `.npy` files
after setting its segmentation model arguments and output directory.

Final prepared VoD layout:

```text
data/VoD/
├── lidar/
├── radar/
├── radar_3frames/
├── radar_5frames/
├── vod_virtual_points/
├── VOD_infos_train.pkl
├── VOD_infos_valid.pkl
├── VOD_infos_trainval.pkl
└── VOD_infos_test.pkl
```

### TJ4DRadSet

Download and extract TJ4DRadSet. Place or link the following directories into
`data/TJ4DRadSet`:

```text
data/TJ4DRadSet/
├── TJ4DRadSet_4DRadar/
│   ├── ImageSets/
│   │   ├── train.txt
│   │   ├── val.txt
│   │   ├── trainval.txt
│   │   └── test.txt
│   └── training/
│       ├── calib/
│       └── velodyne/
├── TJ4DRadSet_Non_Public/
│   └── training/
│       └── image_2/
├── TJ4DRadSet_LiDAR/
└── label_2/
```

The converter uses radar point clouds from
`TJ4DRadSet_4DRadar/training/velodyne`, camera images from
`TJ4DRadSet_Non_Public/training/image_2`, calibration files from
`TJ4DRadSet_4DRadar/training/calib`, and labels from `label_2`.

Example symbolic-link setup:

```bash
mkdir -p data/TJ4DRadSet
ln -s /path/to/TJ4DRadSet/TJ4DRadSet_4DRadar data/TJ4DRadSet/TJ4DRadSet_4DRadar
ln -s /path/to/TJ4DRadSet/TJ4DRadSet_Non_Public data/TJ4DRadSet/TJ4DRadSet_Non_Public
ln -s /path/to/TJ4DRadSet/TJ4DRadSet_LiDAR data/TJ4DRadSet/TJ4DRadSet_LiDAR
ln -s /path/to/TJ4DRadSet/label_2 data/TJ4DRadSet/label_2
```

Generate the TJ4DRadSet info files:

```bash
python projects/RCVAFusion/create_tj4d.py
```

Expected generated files:

```text
data/TJ4DRadSet/
├── TJ4DRadSet_infos_train.pkl
├── TJ4DRadSet_infos_valid.pkl
├── TJ4DRadSet_infos_trainval.pkl
└── TJ4DRadSet_infos_test.pkl
```

Some prepared environments may also contain condition-specific validation info
files:

```text
data/TJ4DRadSet/
├── TJ4DRadSet_infos_dark.pkl
├── TJ4DRadSet_infos_normal.pkl
└── TJ4DRadSet_infos_shiny.pkl
```

These files use the same relative path format as the train/valid/test info
files and can be consumed by configs that point `ann_file` to them.

The TJ4DRadSet config uses:

```python
data_root = 'data/TJ4DRadSet'
data_prefix = dict(
    pts='TJ4DRadSet_4DRadar/training/velodyne',
    img='TJ4DRadSet_Non_Public/training/image_2')
```

For RCVAFusion training, virtual point files are expected at:

```text
data/TJ4DRadSet/tj4d_virtual_points/
```

Each file should be named by sample id, for example:

```text
data/TJ4DRadSet/tj4d_virtual_points/020000.npy
```

The helper script `virtual_points_demo/tj4d_visual_points.py` is configured to
read TJ4DRadSet data from `data/TJ4DRadSet` and can be used to generate these
`.npy` files after setting its segmentation model arguments and output
directory.

Final prepared TJ4DRadSet layout:

```text
data/TJ4DRadSet/
├── TJ4DRadSet_4DRadar/
├── TJ4DRadSet_Non_Public/
├── TJ4DRadSet_LiDAR/
├── label_2/
├── tj4d_virtual_points/
├── TJ4DRadSet_infos_train.pkl
├── TJ4DRadSet_infos_valid.pkl
├── TJ4DRadSet_infos_trainval.pkl
└── TJ4DRadSet_infos_test.pkl
```

## Configs

Main RCVAFusion configs:

```text
projects/RCVAFusion/configs/vod_rcvafusion.py
projects/RCVAFusion/configs/tj4d_rcvafusion.py
```

Both configs import the project plugin:

```python
custom_imports = dict(imports=['projects.RCVAFusion.mmdet3d_plugin'])
```

Make sure commands are run from the repository root so relative paths such as
`data/VoD` and `data/TJ4DRadSet` resolve correctly.

## Quick Checks

Check that the converted info files exist:

```bash
ls data/VoD/VOD_infos_*.pkl
ls data/TJ4DRadSet/TJ4DRadSet_infos_*.pkl
```

Check that the custom ops are importable:

```bash
python - <<'PY'
from projects.RCVAFusion.mmdet3d_plugin.models.sub_models.ops.bev_pool import bev_pool
from projects.RCVAFusion.mmdet3d_plugin.models.sub_models.ops.voxel import voxelize
print('RCVAFusion ops import OK')
PY
```

## Notes

- `projects/RCVAFusion/create_vod.py` and
  `projects/RCVAFusion/create_tj4d.py` should be run from the repository root.
- The generated final info files store point cloud and image file names as
  relative paths. The dataset classes combine them with `data_root` and
  `data_prefix` from the configs.
- If you keep datasets outside this repository, symbolic links under `data/`
  are recommended.
