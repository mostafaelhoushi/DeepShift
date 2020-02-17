from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='unoptimized_cuda',
    ext_modules=[
        CUDAExtension('unoptimized_cuda', [
            'unoptimized_cuda.cpp',
            'unoptimized.cu',
        ],extra_compile_args=['-O3'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })