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

cmake_minimum_required(VERSION 3.20)
project(nvc_endoscopy_tool_tracking_cpp CXX)

find_package(holoscan 3.3 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(nvc_endoscopy_tool_tracking_cpp
  nvc_endoscopy_tool_tracking.cpp
)

target_link_libraries(nvc_endoscopy_tool_tracking_cpp
  PRIVATE
  holoscan::core
  holoscan::ops::format_converter
  holoscan::ops::holoviz
  lstm_tensor_rt_inference
  tool_tracking_postprocessor
  nv_video_decoder
  nv_video_encoder
  nv_video_reader
  tensor_to_file
)

# Copy config file
add_custom_target(nvc_endoscopy_tool_tracking_yaml
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/nvc_endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "nvc_endoscopy_tool_tracking.yaml"
  BYPRODUCTS "nvc_endoscopy_tool_tracking.yaml"
)
add_dependencies(nvc_endoscopy_tool_tracking_cpp nvc_endoscopy_tool_tracking_yaml)

# Add testing
if(BUILD_TESTING)
  add_test(NAME nvc_endoscopy_tool_tracking_cpp_test
           COMMAND nvc_endoscopy_tool_tracking_cpp
                   --config ${CMAKE_CURRENT_BINARY_DIR}/nvc_endoscopy_tool_tracking.yaml
                   --data "${HOLOHUB_DATA_DIR}/endoscopy"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/nvc_endoscopy_tool_tracking)
  set_tests_properties(nvc_endoscopy_tool_tracking_cpp_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Deactivating Graph"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # For aarch64 LD_LIBRARY_PATH needs to be set
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL aarch64 OR CMAKE_SYSTEM_PROCESSOR STREQUAL arm64)
    set_tests_properties(nvc_endoscopy_tool_tracking_cpp_test PROPERTIES ENVIRONMENT
                    "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra/")
  endif()
endif()

# Set install rules
install(TARGETS nvc_endoscopy_tool_tracking_cpp
        DESTINATION "bin"
        COMPONENT "holoscan-examples")

install(FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/nvc_endoscopy_tool_tracking.yaml"
        DESTINATION "share/holohub/applications/nvidia_video_codec/nvc_endoscopy_tool_tracking/"
        COMPONENT "holoscan-examples")

