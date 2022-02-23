#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

setup(
    name="cortical-breach-detection",
    version="0.0.0",
    description="Detecing cortical breaches.",
    author="Benjamin D. Killeen",
    author_email="killeen@jhu.edu",
    url="https://github.com/benjamindkilleen/cortical_breach_detection",
    install_requires=[
        "torch",
        "torchvision",
        "pytorch-lightning",
        "hydra-core",
        "omegaconf",
        "rich",
        "numpy",
        "deepdrr",
    ],
    packages=find_packages(),
)
