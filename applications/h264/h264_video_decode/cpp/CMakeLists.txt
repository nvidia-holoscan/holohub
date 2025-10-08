# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(h264_video_decode CXX)

find_package(holoscan 3.6.1 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(h264_video_decode
  main.cpp
)

target_link_libraries(h264_video_decode
  PRIVATE
  holoscan::core
  holoscan::ops::gxf_codelet
  holoscan::ops::format_converter
  holoscan::ops::holoviz
)

# Copy config file
add_custom_target(h264_video_decode_yaml
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/h264_video_decode.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "h264_video_decode.yaml"
  BYPRODUCTS "h264_video_decode.yaml"
)
add_dependencies(h264_video_decode h264_video_decode_yaml)

# Download the endoscopy sample data
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL nvidia/clara-holoscan/holoscan_endoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
  )
  add_dependencies(h264_video_decode endoscopy_data)
endif()

# Add testing
if(BUILD_TESTING)
  add_test(NAME h264_video_decode_cpp_test
           COMMAND h264_video_decode h264_video_decode.yaml
                   --data "${HOLOHUB_DATA_DIR}/endoscopy"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(h264_video_decode_cpp_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Waiting for completion"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # For aarch64 LD_LIBRARY_PATH needs to be set
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
    set_tests_properties(h264_video_decode_cpp_test PROPERTIES ENVIRONMENT
                    "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:/usr/lib/aarch64-linux-gnu/tegra/")
  endif()
endif()
