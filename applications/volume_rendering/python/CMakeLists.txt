# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.20)
project(volume_rendering_python LANGUAGES NONE)

find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable the operators
add_library(volume_rendering_python INTERFACE)
target_link_libraries(volume_rendering_python INTERFACE holoscan::core holoscan::ops::holoviz holoscan::ops::volume_loader holoscan::ops::volume_renderer)

# Download the volume rendering sample data
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(volume_rendering
    URL nvidia/clara-holoscan/holoscan_volume_rendering_sample_data:20230628
    DOWNLOAD_NAME holoscan_volume_rendering_python_sample_data_20230628.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )
  add_dependencies(volume_rendering_python volume_rendering_data)
endif()

if(BUILD_TESTING)
  add_test(NAME volume_rendering_python_test
           COMMAND ${PYTHON_EXECUTABLE_TEST} ${CMAKE_CURRENT_SOURCE_DIR}/volume_rendering.py
             --count=100
             -c "${HOLOHUB_DATA_DIR}/volume_rendering/config.json"
             -d "${HOLOHUB_DATA_DIR}/volume_rendering/highResCT.mhd"
             -m "${HOLOHUB_DATA_DIR}/volume_rendering/smoothmasks.seg.mhd"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_property(TEST volume_rendering_python_test PROPERTY ENVIRONMENT
           "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(volume_rendering_python_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Application has finished running.")
endif()
