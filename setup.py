from setuptools import setup, find_packages
import sys
import subprocess


# Detecting if pytorch with or without CUDA support should be installed
try:
    subprocess.check_output('nvidia-smi')
    HAS_NVIDIA = True
except:
    HAS_NVIDIA = False

if HAS_NVIDIA:
    dependency_links = []
else:
    dependency_links = ['https://download.pytorch.org/whl/cpu']
    print("torch_spex setup info: Did not find NVIDIA card defaulting to CPU-only installation")

setup(
    name='torch_spex',
    packages = find_packages(),
    install_requires=[
        'sphericart[torch]',
        'numpy',
        'ase',
        'torch',
        'scipy',
        'equistore-operations@git+https://github.com/Luthaf/equistore.git@torch-core-classes#subdirectory=python/equistore-operations',
        'equistore-torch@git+https://github.com/Luthaf/equistore.git@torch-core-classes#subdirectory=python/equistore-torch',
    ],
    dependency_links = dependency_links
)
