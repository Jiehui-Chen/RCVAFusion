# RCVAFusion
## Abstract

4D millimeter-wave radar plays a critical role in object detection for
autonomous driving and robotics under all-weather and all-lighting conditions.
Recently, virtual-point-based approaches have attracted widespread attention due
to their ability to address radar data sparsity by complementing the depth of
image instance points with the nearest 3D points. However, existing
radar-camera fusion methods based on virtual points simply incorporate virtual
points into the raw radar points as a form of data augmentation, overlooking the
potential of virtual points that have an inherent association with both radar
and images. To address these issues, we present a novel radar-camera fusion
network, RCVAFusion, for 3D object detection. Specifically, we first design an
association branch that employs Object Area Sampling (OAS) and Virtual-Raw
Points Depth Lifting (VRPDL). This branch facilitates a deep interaction between
radar geometric features and image semantic features through the medium of
virtual points to generate an association feature. Then, we introduce the
Dual-step Feature Aggregation (DFA) to promote feature fusion from radar, image,
and association branches by establishing aggregation priorities based on feature
similarity in two steps. Experimental results on the TJ4DRadSet and
View-of-Delft (VoD) datasets demonstrate that our method efficiently fuses radar
and camera through virtual points and achieves state-of-the-art performance.

## Result Visualization

[Watch the result visualization video](docs/ouput.mp4)

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
cd projects/RCVAFusion
python setup.py build_ext --inplace
cd ../..
```



## Data Preparation


1、Download and extract the official View-of-Delft public dataset. Place or link the following directories into `data/VoD`:

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


2、Download and extract TJ4DRadSet. Place or link the following directories into
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

3、You can download the virtual radar points and raw points from [Baidu](https://pan.baidu.com/s/1TK81mvDOqM-Ep3ktstdT5g?pwd=r483) and unzip them to the dataset folder.

4、(Optional) Or you can choose to generate virtual radar points by yourself following [here](virtual_points_demo/README.md).

5、You can download the Dataset info files from [Baidu](https://pan.baidu.com/s/1cnKQiCxR8jmRxs80pZJmaA?pwd=jn6m) or generate them following:

```bash
python projects/RCVAFusion/create_vod.py
python projects/RCVAFusion/create_tj4d.py
```

6、Final prepared data layout:

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
## Model Zoo

We retrained the model and achieved better performance compared to the results
reported in the tables of the paper. We provide the checkpoints on
View-of-Delft (VoD) and TJ4DRadSet datasets, reproduced with the released
codebase.

| Dataset | Backbone | EAA 3D mAP | DC 3D mAP | Model Weights |
| --- | --- | ---: | ---: | --- |
| [View-of-Delft](https://tudelft-iv.github.io/view-of-delft-dataset/) | ResNet50 | 62.85 | 82.32 | [Link](https://pan.baidu.com/s/1mvwh3tI3d450nZ7vwQ-p0A?pwd=c7av) |
| [TJ4DRadSet](https://pan.baidu.com/s/1PmTIOtQBLAIICEAPM4-fOg?pwd=g7na) | ResNet50 | 42.05 | 49.35 | [Link]() |

## Training and Evaluating
1、We provide pre-trained image backbone models from [baidu](https://pan.baidu.com/s/1BoEuRNyYMtBhUKW-hYmDEQ?pwd=9g88) .

2、train RCVAFusion with 4 GPUs:
```bash
bash tools/dist_train.sh projects/RCVAFusion/configs/vod_rcvafusion.py 4
bash tools/dist_train.sh projects/RCVAFusion/configs/tj4d_rcvafusion.py 4
```

3、test RCVAFusion with single GPU:

```bash
python tools/test.py projects/RCVAFusion/configs/vod_rcvafusion.py ckpts/vod_rcvafusion.pth
python tools/test.py projects/RCVAFusion/configs/tj4d_rcvafusion.py ckpts/tj4d_rcvafusion.pth
```

## Acknowledgements
Many thanks to the open-source repositories:

[mmdetection3d](https://github.com/open-mmlab/mmdetection3d)

[mask2former](https://github.com/facebookresearch/Mask2Former)

[SGDet3D](https://github.com/shawnnnkb/SGDet3D)

[HGSFusion](https://github.com/garfield-cpp/HGSFusion)
