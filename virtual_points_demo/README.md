# Generate virtual radar points

This tutorial will guide you on how to generate virtual radar points on VoD and
TJ4DRadSet.

## Installation

1. The environment is the same as
   [Mask2Former](https://github.com/facebookresearch/Mask2Former).

- Linux or macOS with Python >= 3.6
- PyTorch >= 1.7 and torchvision that matches the PyTorch installation.
  Install them together at pytorch.org to make sure of this. Note, please
  check PyTorch version matches that is required by Detectron2.
- Detectron2: follow Detectron2 installation instructions.
- OpenCV is optional but needed by demo and visualization.
- Install Python requirements:

```bash
pip install -r requirements.txt
```

2. Copy `vod_visual_points.py` and `tj4d_visual_points.py` to the Mask2Former
   `demo/` directory.

3. Download the segmentation weight from
   [here](https://pan.baidu.com/s/1BoEuRNyYMtBhUKW-hYmDEQ?pwd=9g88), and place
   it at `./ckpts/`.

## Usage

Run the scripts from the Mask2Former project root after preparing the dataset
folders in this repository.

If you are aiming for a certain level of inference speed, the Swin-T (configs/cityscapes/instance-segmentation/swin/maskformer2_swin_tiny_bs16_90k.yaml) model is recommended.

If you require higher precision, the Swin-L (configs/cityscapes/instance-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_90k.yaml) model is recommended.

For View-of-Delft:

```bash
python demo/vod_visual_points.py \
  --config-file configs/cityscapes/instance-segmentation/swin/your_choose.yaml \
  --pts-save-path /path/to/mmdetection3d/data/VoD/vod_virtual_points \
  --opts MODEL.WEIGHTS ./ckpts/your_choose.pth
```

For TJ4DRadSet:

```bash
python demo/tj4d_visual_points.py \
  --config-file configs/cityscapes/instance-segmentation/swin/your_choose.yaml \
  --pts-save-path /path/to/mmdetection3d/data/TJ4DRadSet/tj4d_virtual_points \
  --opts MODEL.WEIGHTS ./ckpts/your_choose.pth
```

The output files are `.npy` files named by sample id, for example:

```text
data/VoD/vod_virtual_points/00001.npy
data/TJ4DRadSet/tj4d_virtual_points/020000.npy
```
