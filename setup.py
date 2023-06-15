#   coding:utf-8
#   This file is part of MF_Fusion.

__author__ = 'Guanjie Wang'
__email__ = "gjwang.buaa@gmail.edu.cn"
__version__ = 1.0
__init_date__ = '2023/06/15 10:16:04'
__maintainer__ = 'Guanjie Wang'
__update_date__ = '2023/06/15 10:16:04'

import os
from setuptools import find_packages, setup


NAME = 'mffusion'
VERSION = '0.1.0-beta1'
DESCRIPTION = 'Multi-fidelity fusion toolbox of most GP-based mtehod'
README_FILE = os.path.join(os.path.dirname(__file__), 'README.md')
LONG_DESCRIPTION = open(README_FILE, encoding='UTF8').read()

REQUIREMENTS = ['numpy', 'torch', 'scikit-learn', 'scipy', 'tensorly', 'yaml', 'h5py']
URL = "https://github.com/IceLab-X/MF-Fusion"
AUTHOR = 'Wei Xing, Zen Xingle '
AUTHOR_EMAIL = ''
LICENSE = 'MIT'
PACKAGES = find_packages()
PACKAGE_DATA = {}
ENTRY_POINTS = {}


def setup_package():
    setup(
        name=NAME,
        version=VERSION,
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        long_description_content_type="text/markdown",
        author=AUTHOR,
        author_email=AUTHOR_EMAIL,
        license=LICENSE,
        packages=find_packages(),
        package_data=PACKAGE_DATA,
        include_package_data=True,
        entry_points=ENTRY_POINTS,
        install_requires=REQUIREMENTS,
        cmdclass={},
        zip_safe=False,
        url=URL
    )


if __name__ == '__main__':
    setup_package()
