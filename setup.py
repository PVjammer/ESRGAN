import os
import sys
from setuptools import setup, find_packages

pkg_name = "esrgan"


setup(name=pkg_name,
        package_dir={"":"src",},
        version="0.1",
        description="Installable package for ESRGAN",
        url="https://github.com/PVjammer/ESRGAN/esrgan",
        license="Apache License, Version 2.0",
        packages=find_packages(),
        install_requires=[
            "setuptools>=40.3.0",
            "requests",
            "numpy",
            "opencv-python",
            "torch"
            ],

#        data_files=[("models", ["src/esrgan/models/RRDB_ESRGAN_x4.pth"])],

        py_modules=[
            "esrgan.rrdbnet",
            "esrgan.transer"
            ]
        )




