#!/usr/bin/env python
import os
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension


with open('requirements.txt') as f:
    requirements = f.read().splitlines()


long_description = """
This package implements a Torch extension that is an efficient CUDA
and C++ parallel algorithm for an operator that samples from a
categorical distribution passed in as a tensor of un-normalized log
probabilities.  The output will be a one-hot vector most of the time;
and some of the time it will be a value interpolated between two
one-hot vectors.

We call it "flow sampling" because the derivation of the algorithm
(particularly the derivative computation) invokes fluid flow.

[TODO]: webpage, etc.
"""


def configure_extensions():
    out = [
        CppExtension(
            'torch_flow_sampling_cpu',
            [
                os.path.join('torch_flow_sampling', 'flow_sampling_cpu.cpp'),
            ],
        )
    ]
    try:
        out.append(
            CUDAExtension(
                'torch_flow_sampling_cuda',
                [
                    os.path.join('torch_flow_sampling', 'flow_sampling_cuda.cpp'),
                    os.path.join('torch_flow_sampling', 'flow_sampling_cuda_kernel.cu'),
                ],
            )
        )
    except Exception as e:
        print(f'Failed to build CUDA extension, this part of the package will not work. Reason: {str(e)}')
    return out


setup(
    name='torch_flow_sampling',
    version='1.0.0',
    description='Differentiable sampling of a categorical variable',
    long_description=long_description,
    long_description_content_type='text/markdown',
    install_requires=requirements,
    python_requires='>=3.6',
    packages=find_packages(),
    author='Dan Povey',
    license='BSD',
    ext_modules=configure_extensions(),
    cmdclass={
        'build_ext': BuildExtension
    },
    keywords=[
        'pytorch', 'sampling', 'categorical'
    ],
)
