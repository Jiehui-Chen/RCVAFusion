#!/usr/bin/env bash
# Setup the conda environment for mmdetection3d

# set versions of mmengine, mmcv and mmdet
mmengine_version="0.9.0"
mmcv_version="2.0.1"
mmdet_version="3.1.0"

echo "Please choose the installation methods of mmengine, mmcv and mmdet:"
echo "1. Install from mim (for most users)"
echo "2. Install from source code"

read choice

#check CUDA version == 11.7
nvcc -V

# remove old conda environment
conda remove -n openmmlab --all -y

# create a conda environment and activate it
conda create -n openmmlab python=3.8 -y
eval "$(conda shell.bash hook)" # Solve the problem of conda command not found
conda activate openmmlab

# install pytorch 2.0.1
# CUDA 11.7
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

pip install -U openmim -i https://pypi.tuna.tsinghua.edu.cn/simple

if [ $choice -eq 1 ]
then
    # install mmengine, mmcv and mmdet using mim
    mim install "mmengine==$mmengine_version" -i https://pypi.tuna.tsinghua.edu.cn/simple
    mim install "mmcv==$mmcv_version" -i https://pypi.tuna.tsinghua.edu.cn/simple
    mim install "mmdet==$mmdet_version" -i https://pypi.tuna.tsinghua.edu.cn/simple

elif [ $choice -eq 2 ]
then
    # install mmengine from source
    git clone -b v$mmengine_version https://github.com/open-mmlab/mmengine.git
    cd mmengine
    pip install -e . -v -i https://pypi.tuna.tsinghua.edu.cn/simple
    cd ..

    # install mmcv from source
    git clone -b v$mmcv_version https://github.com/open-mmlab/mmcv.git
    cd mmcv
    pip install -r requirements/optional.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
    # Check the nvcc version (requires 9.2+. Skip if no GPU available.)
    nvcc --version
    # Check the gcc version (requires 5.4+)
    gcc --version
    pip install -e . -v -i https://pypi.tuna.tsinghua.edu.cn/simple
    python .dev_scripts/check_installation.py
    cd ..

    # install mmdet from source
    git clone -b v$mmdet_version https://github.com/open-mmlab/mmdetection.git
    cd mmdetection
    pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple
    cd ..
else
    echo "Invalid choice"
fi

# install mmdetection3d for develop
pip install -v -e . -i https://pypi.tuna.tsinghua.edu.cn/simple

# install spconv 2.0+
pip install spconv-cu117 -i https://pypi.tuna.tsinghua.edu.cn/simple

# install pre-commit
pip install -U pre-commit -i https://pypi.tuna.tsinghua.edu.cn/simple
pre-commit install

# verify the installation
mim download mmdet3d --config pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car --dest demo/data/kitti/
rm demo/data/kitti/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py # delete redundant config file
python demo/pcd_demo.py demo/data/kitti/000008.bin configs/pointpillars/pointpillars_hv_secfpn_8xb6-160e_kitti-3d-car.py demo/data/kitti/hv_pointpillars_secfpn_6x8_160e_kitti-3d-car_20220331_134606-d42d15ed.pth --show
