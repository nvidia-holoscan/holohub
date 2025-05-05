# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(volume_rendering)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(volume_rendering
  json_loader.cpp
  volume_rendering.cpp
)

target_link_libraries(volume_rendering
  holoscan::core
  holoscan::ops::volume_loader
  holoscan::ops::volume_renderer
  holoscan::ops::holoviz
  nlohmann_json::nlohmann_json
)

# Download the volume rendering sample data
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(volume_rendering
    URL nvidia/clara-holoscan/holoscan_volume_rendering_sample_data:20230628
    DOWNLOAD_NAME holoscan_volume_rendering_sample_data_20230628.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
  )
  add_dependencies(volume_rendering volume_rendering_data)
endif()

if(BUILD_TESTING)
  # Add test
  add_test(NAME volume_rendering_test
           COMMAND volume_rendering
             --count=100
             -c "${HOLOHUB_DATA_DIR}/volume_rendering/config.json"
             -d "${HOLOHUB_DATA_DIR}/volume_rendering/highResCT.mhd"
             -m "${HOLOHUB_DATA_DIR}/volume_rendering/smoothmasks.seg.mhd"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(volume_rendering_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Application has finished running.")
endif()
