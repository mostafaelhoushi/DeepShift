from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='shift_kernal', ext_modules=[cpp_extension.CppExtension('shift_kernal', ['shift_kernal.cpp'], extra_compile_args=['-fopenmp', '-O3'])], cmdclass={'build_ext': cpp_extension.BuildExtension})

