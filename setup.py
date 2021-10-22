#!/usr/bin/env python
import os
import pathlib
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

with open('requirements.txt') as f:
    requirements = f.read().splitlines()


long_description = """
This package implements a Torch extension that is an efficient CUDA
and C++ parallel algorithm for an operator that samples multiple
distinct samples, with associated weights, from a categorical distribution,
outputting a discretized version of the categorical distribution whose expected
value equals the input, and which gets closer to it the more
samples you use.   This is done in such a way that it's straightforward
to model the probability of the output distribution.

[TODO]: webpage, etc.
"""


def configure_extensions():
    out = [
        CppExtension(
            'torch_iterative_sampling_cpu',
            [
                os.path.join('torch_iterative_sampling', 'iterative_sampling_cpu.cpp'),
            ],
        )
    ]
    try:
        # this_dir is the directory where this setup.py is located.
        this_dir = pathlib.Path(__file__).parent.resolve()
        out.append(
            CUDAExtension(
                'torch_iterative_sampling_cuda',
                [
                    os.path.join('torch_iterative_sampling', 'iterative_sampling_cuda.cpp'),
                    os.path.join('torch_iterative_sampling', 'iterative_sampling_cuda_kernel.cu'),
                ],
                extra_compile_args={'cxx': [], 'nvcc': [f'-I{this_dir}/cub']}
            )
        )
    except Exception as e:
        print(f'Failed to build CUDA extension, this part of the package will not work. Reason: {str(e)}')
    return out


setup(
    name='torch_iterative_sampling',
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
