#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from pathlib import Path
from setuptools import setup

# Get version
from cli._version import __version__ as version

# Get relative path to this file from where it is called
file_dir = Path(__file__).parent.absolute()
work_dir = Path(os.getcwd())
relative_path = file_dir.relative_to(work_dir)

# setup wheel
setup(
    name='holohub-cli',
    version=version,
    description='Holohub CLI',
    author='NVIDIA',
    author_email='agirault@nvidia.com',
    url='https://github.com/nvidia-holoscan/holohub',
    packages=['holohub_cli', 'holohub_metadata'],
    package_dir={
        'holohub_cli': relative_path / 'cli',
        'holohub_metadata': relative_path / 'metadata',
    },
    include_package_data=True,
    package_data={
        'holohub_cli': ['*.py'],
        'holohub_metadata': ['*.py', '*.json'],
    },
    entry_points={
        'console_scripts': [
            'holohub=holohub_cli.holohub:main',
        ],
    },
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
    ],
    python_requires='>=3.8',
)
