from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name='shift_kernal', ext_modules=[cpp_extension.CppExtension('shift_kernal', ['shift_kernal.cpp'])], cmdclass={'build_ext': cpp_extension.BuildExtension})

