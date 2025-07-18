# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

find_package(holoscan 3.3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


add_executable(slang_simple
    slang_simple.cpp
)

target_link_libraries(slang_simple
  PRIVATE
    holoscan::core
    holoscan::ops::slang_shader
)

# Copy shader file
add_custom_target(slang_simple_shader_file
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/simple.slang" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/simple.slang"
)
add_dependencies(slang_simple slang_simple_shader_file)


if(BUILD_TESTING)
  add_test(NAME slang_simple_test
           COMMAND slang_simple
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(slang_simple_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Received value: 20"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed;error")
endif()

