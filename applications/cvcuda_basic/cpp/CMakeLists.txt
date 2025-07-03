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
project(cvcuda_basic CXX)

find_package(holoscan 3.4 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
# find_package(CUDAToolkit REQUIRED)  # Holoscan will already have included CUDA
find_package(nvcv_types REQUIRED)
find_package(cvcuda REQUIRED)

add_executable(cvcuda_basic
  main.cpp
)

target_link_libraries(cvcuda_basic
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::holoviz
  holoscan::ops::cvcuda_holoscan_interop
  holoscan::data_loggers::message_logger
  nvcv_types
  cvcuda
)

# Copy the config to the binary directory
add_custom_target(cvcuda_basic_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/cvcuda_basic.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "cvcuda_basic.yaml"
  BYPRODUCTS "cvcuda_basic.yaml"
)
add_dependencies(cvcuda_basic cvcuda_basic_deps)

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file to only play 10 frames
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/cvcuda_basic.yaml" CONFIG_FILE)
  string(REGEX REPLACE "source:[^\n]*" "source: replayer" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cvcuda_basic_testing.yaml" ${CONFIG_FILE})

  # Add test
  add_test(NAME cvcuda_basic_cpp_test
           COMMAND cvcuda_basic ${CMAKE_CURRENT_BINARY_DIR}/cvcuda_basic_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/endoscopy"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(cvcuda_basic_cpp_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
