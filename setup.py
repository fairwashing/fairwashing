#!/usr/bin/env python3
from setuptools import setup, find_packages

setup(
    name="fairwasher",
    use_scm_version=True,
    packages=find_packages(),
    install_requires=[
        'torch',
        'torchvision',
        'numpy',
        'tqdm',
        'h5py',
        'scikit-image',
        'click',
        'scipy',
        'tensorboard',
    ],
    setup_requires=[
        'setuptools_scm',
    ],
)
