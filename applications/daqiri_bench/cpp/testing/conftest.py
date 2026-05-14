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

import pytest


def pytest_addoption(parser):
    """Add command line options to pytest."""
    parser.addoption(
        "--workdir",
        action="store",
        default=os.getcwd(),
        help="Working directory for tests (where executables and config files are located)",
    )


@pytest.fixture(scope="module")
def work_dir(request):
    """Return the working directory."""
    return request.config.getoption("--workdir")
