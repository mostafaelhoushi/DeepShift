from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='shift_kernel', ext_modules=[cpp_extension.CppExtension('shift_kernel', ['shift_kernel.cpp'], extra_compile_args=['-fopenmp', '-O3'])], cmdclass={'build_ext': cpp_extension.BuildExtension})

