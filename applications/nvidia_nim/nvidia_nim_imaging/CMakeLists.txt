# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(nvidia_nim_imaging LANGUAGES NONE)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


# Enable the operators
add_library(nvidia_nim_imaging INTERFACE)
target_link_libraries(nvidia_nim_imaging INTERFACE holoscan::core)

# Download the volume rendering sample data
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(nvidia_nim_imaging
    URL nvidia/clara-holoscan/holoscan_volume_rendering_sample_data:20230628
    DOWNLOAD_NAME holoscan_volume_rendering_sample_data_20230628.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )
  add_dependencies(nvidia_nim_imaging nvidia_nim_imaging_data)
endif()