#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2023 Apple Inc. All Rights Reserved.
#

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='clustencuda',
    version='0.1',
    author='Ziwen Chen',
    author_email='chenziw@oregonstate.edu',
    description='Cluster Attention CUDA Kernel',
    ext_modules=[
        CUDAExtension('clustenqk_cuda', [
            'clustenqk_cuda.cpp',
            'clustenqk_cuda_kernel.cu',
        ]),
        CUDAExtension('clustenav_cuda', [
            'clustenav_cuda.cpp',
            'clustenav_cuda_kernel.cu',
        ]),
        CUDAExtension('clustenwf_cuda', [
            'clustenwf_cuda.cpp',
            'clustenwf_cuda_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
