# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(xr_gsplat NONE)

find_package(holoscan 3.3 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Install gsplat
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_target(install_gsplat ALL)
add_custom_command(
    TARGET install_gsplat
    COMMAND ${Python3_EXECUTABLE} -m pip install --no-build-isolation git+https://github.com/nerfstudio-project/gsplat.git
    COMMENT "Installing gsplat..."
)

if(BUILD_TESTING)
  add_test(NAME xr_gsplat_checkpoint_test
           COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                   "${CMAKE_CURRENT_SOURCE_DIR}/test_gsplat_loader_op.py"
           WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  set_tests_properties(xr_gsplat_checkpoint_test PROPERTIES
                       LABELS "application;python;unit"
                       TIMEOUT 120)
endif()
