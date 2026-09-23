# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(multi_device_inference LANGUAGES CXX CUDA)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(multi_device_inference main.cpp)

target_link_libraries(multi_device_inference
  PRIVATE
    holoscan::core
    tensorrt_multi_device_inference
)
set_property(TARGET multi_device_inference PROPERTY CUDA_ARCHITECTURES "native")

# Copy the config next to the binary
add_custom_target(multi_device_inference_yaml
  COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_SOURCE_DIR}/multi_device_inference.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/multi_device_inference.yaml"
)
add_dependencies(multi_device_inference multi_device_inference_yaml)

if(BUILD_TESTING)
  # Requires >= 2 GPUs and the sharded engine plans (data/multi_device_inference/).
  # Labelled "multigpu" so single-GPU CI can exclude it (see ctest -LE multigpu).
  add_test(NAME multi_device_inference_cpp_test
           COMMAND multi_device_inference "${CMAKE_CURRENT_SOURCE_DIR}/multi_device_inference.yaml"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(multi_device_inference_cpp_test PROPERTIES
    LABELS "multigpu"
    PASS_REGULAR_EXPRESSION "Multi-Device inference complete"
    FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()

install(TARGETS multi_device_inference)
