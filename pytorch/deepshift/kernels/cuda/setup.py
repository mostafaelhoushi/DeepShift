from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='deepshift_cuda',
    ext_modules=[
        CUDAExtension('deepshift_cuda', [
            'shift_cuda.cpp',
            'shift.cu',
        ],extra_compile_args=['-O3'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })

setup(
    name='unoptimized_cuda_kernel',
    ext_modules=[
        CUDAExtension('unoptimized_cuda_kernel', [
            'unoptimized_cuda.cpp',
            'unoptimized_cuda_kernel.cu',
        ],extra_compile_args=['-O3'])
    ],
    cmdclass={
        'build_ext': BuildExtension
    })