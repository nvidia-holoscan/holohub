# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(h264_endoscopy_tool_tracking CXX)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(h264_endoscopy_tool_tracking
  main.cpp
)

target_link_libraries(h264_endoscopy_tool_tracking
  PRIVATE
  holoscan::core
  holoscan::ops::format_converter
  holoscan::ops::holoviz
  holoscan::ops::video_decoder
  holoscan::ops::video_encoder
  holoscan::ops::video_read_bitstream
  holoscan::ops::video_write_bitstream
  holoscan::ops::tensor_to_video_buffer
  lstm_tensor_rt_inference
  tool_tracking_postprocessor
)

# Copy config file
add_custom_target(h264_endoscopy_tool_tracking_yaml
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/h264_endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "h264_endoscopy_tool_tracking.yaml"
  BYPRODUCTS "h264_endoscopy_tool_tracking.yaml"
)
add_dependencies(h264_endoscopy_tool_tracking h264_endoscopy_tool_tracking_yaml)

# Download the endoscopy sample data
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL https://api.ngc.nvidia.com/v2/resources/nvidia/clara-holoscan/holoscan_endoscopy_sample_data/versions/20230222/zip
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    URL_MD5 d54f84a562d29ed560a87d2607eba973
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
  )
  add_dependencies(h264_endoscopy_tool_tracking endoscopy_data)
endif()

# Add testing
if(BUILD_TESTING)
  add_test(NAME h264_endoscopy_tool_tracking_cpp_test
           COMMAND h264_endoscopy_tool_tracking h264_endoscopy_tool_tracking.yaml
                   --data "${HOLOHUB_DATA_DIR}/endoscopy"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(h264_endoscopy_tool_tracking_cpp_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Deactivating Graph"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # For aarch64 LD_LIBRARY_PATH needs to be set
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
    set_tests_properties(h264_endoscopy_tool_tracking_cpp_test PROPERTIES ENVIRONMENT
                    "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:/usr/lib/aarch64-linux-gnu/tegra/")
  endif()
endif()
