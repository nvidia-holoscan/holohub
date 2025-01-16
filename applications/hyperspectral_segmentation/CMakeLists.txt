# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(holoscan 0.6 REQUIRED CONFIG
               PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the hyperspectral sample data
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(hyperspectral
    URL nvidia/clara-holoscan/hyperspectral_segmentation:20231109
    DOWNLOAD_NAME hyperspectral_segmentation_20231109.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )
endif()

if(BUILD_TESTING)
  # Add test
  add_test(NAME hyperspectral_segmentation_test
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/hyperspectral_segmentation.py
                   --output_folder ${CMAKE_CURRENT_SOURCE_DIR}/
                   --data "${HOLOHUB_DATA_DIR}/hyperspectral/data"
                   --model "${HOLOHUB_DATA_DIR}/hyperspectral"
                   --count 10
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(hyperspectral_segmentation_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Application finished successfully")
endif()
