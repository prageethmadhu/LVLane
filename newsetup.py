from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='nms_impl',
    ext_modules=[
        CppExtension('nms_impl', ['nms.cpp']),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
