from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='deepshift_cpu', 
    ext_modules=[
        cpp_extension.CppExtension('deepshift_cpu', [
            'shift_cpu.cpp'
        ], extra_compile_args=['-fopenmp', '-O3'])
    ], 
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    })

