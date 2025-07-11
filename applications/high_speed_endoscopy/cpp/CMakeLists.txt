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
project(high_speed_endoscopy CXX)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(high_speed_endoscopy
  main.cpp
)

target_link_libraries(high_speed_endoscopy
  PRIVATE
  holoscan::core
  holoscan::ops::bayer_demosaic
  holoscan::ops::holoviz
  emergent_source
)

target_include_directories(high_speed_endoscopy
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../operators/emergent_source
)

target_compile_definitions(high_speed_endoscopy
PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR})

# Copy config file
add_custom_target(high_speed_endoscopy_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/high_speed_endoscopy.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/high_speed_endoscopy.yaml"
)
add_dependencies(high_speed_endoscopy high_speed_endoscopy_yaml)

# Add testing
if(BUILD_TESTING)
  # Warning: this test just check that the application starts if there is no camera
  add_test(NAME high_speed_endoscopy_test_cpp
           COMMAND high_speed_endoscopy ${CMAKE_CURRENT_BINARY_DIR}/high_speed_endoscopy.yaml
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(high_speed_endoscopy_test_cpp PROPERTIES
                       PASS_REGULAR_EXPRESSION "No EVT camera found.")
endif()
