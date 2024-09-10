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

find_package(holoscan 2.1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable the operators
add_library(h264_endoscopy_tool_tracking_python INTERFACE)
target_link_libraries(h264_endoscopy_tool_tracking_python INTERFACE holoscan::core holoscan::ops::holoviz holoscan::ops::lstm_tensor_rt_inference holoscan::ops::tensor_to_video_buffer holoscan::ops::tool_tracking_postprocessor)

# Add testing
if(BUILD_TESTING)
  add_test(NAME h264_endoscopy_tool_tracking_python_test
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/h264_endoscopy_tool_tracking.py
                   --config ${CMAKE_CURRENT_SOURCE_DIR}/h264_endoscopy_tool_tracking.yaml
                   --data ${HOLOHUB_DATA_DIR}/endoscopy
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_property(TEST h264_endoscopy_tool_tracking_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(h264_endoscopy_tool_tracking_python_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Deactivating Graph"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # For aarch64 LD_LIBRARY_PATH needs to be set
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
    set_tests_properties(h264_endoscopy_tool_tracking_python_test PROPERTIES ENVIRONMENT
                    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra/")
  endif()
endif()
