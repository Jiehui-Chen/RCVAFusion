import os

from setuptools import setup

import torch
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


def make_cuda_ext(name,
                  module,
                  sources,
                  sources_cuda=None,
                  extra_args=None,
                  extra_include_path=None):
    """Create a PyTorch C++/CUDA extension for RCVAFusion ops."""
    sources_cuda = sources_cuda or []
    extra_args = extra_args or []
    extra_include_path = extra_include_path or []

    define_macros = []
    extra_compile_args = {'cxx': extra_args}

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [('WITH_CUDA', None)]
        extension = CUDAExtension
        extra_compile_args['nvcc'] = extra_args + [
            '-D__CUDA_NO_HALF_OPERATORS__',
            '-D__CUDA_NO_HALF_CONVERSIONS__',
            '-D__CUDA_NO_HALF2_OPERATORS__',
            '-gencode=arch=compute_70,code=sm_70',
            '-gencode=arch=compute_75,code=sm_75',
            '-gencode=arch=compute_80,code=sm_80',
            '-gencode=arch=compute_86,code=sm_86',
        ]
        sources += sources_cuda
    else:
        print(f'Compiling {name} without CUDA')
        extension = CppExtension

    return extension(
        name=f'{module}.{name}',
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        include_dirs=extra_include_path,
        define_macros=define_macros,
        extra_compile_args=extra_compile_args)


if __name__ == '__main__':
    setup(
        name='rcvafusion',
        version='0.0.1',
        description='RCVAFusion project extensions',
        ext_modules=[
            make_cuda_ext(
                name='bev_pool_ext',
                module=('projects.RCVAFusion.mmdet3d_plugin.models.'
                        'sub_models.ops.bev_pool'),
                sources=[
                    'src/bev_pool.cpp',
                    'src/bev_pool_cuda.cu',
                ]),
            make_cuda_ext(
                name='voxel_layer',
                module=('projects.RCVAFusion.mmdet3d_plugin.models.'
                        'sub_models.ops.voxel'),
                sources=[
                    'src/voxelization.cpp',
                    'src/scatter_points_cpu.cpp',
                    'src/scatter_points_cuda.cu',
                    'src/voxelization_cpu.cpp',
                    'src/voxelization_cuda.cu',
                ]),
        ],
        cmdclass={'build_ext': BuildExtension},
        zip_safe=False)
