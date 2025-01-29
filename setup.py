import glob
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

def get_extensions():
    # Use only C++ files for CPU
    op_files = glob.glob('./lanedet/ops/csrc/*.c*')  # Adjust this path if necessary
    ext_name = 'lanedet.ops.nms_impl'

    ext_ops = CppExtension(
        name=ext_name,
        sources=op_files,
        extra_compile_args={'cxx': []},  # Add additional compiler flags here if needed
    )

    return [ext_ops]

setup(
    name='lanedet',
    version='1.0',
    description='Lane Detection open toolbox via PyTorch (CPU version)',
    author='Your Name',
    author_email='your_email@example.com',
    url='https://github.com/your-repo/lvlane',  # Update this with your repository URL
    packages=['lanedet'],  # Specify your package directory here
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},
    install_requires=[
        'torch>=1.8.0',  # Ensure torch is installed
        'torchvision>=0.9.0',
        'scikit-learn',
        'numpy',
        'pandas',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Specify your Python version requirement
)
