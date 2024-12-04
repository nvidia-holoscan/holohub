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
project(simple_radar_pipeline CXX)

find_package(holoscan 2.6.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Enable cuda language
set(CMAKE_CUDA_ARCHITECTURES "70;80")
enable_language(CUDA)

# Download MatX
include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG v0.9.0
)
FetchContent_MakeAvailable(MatX)

add_executable(simple_radar_pipeline
  simple_radar_pipeline.cu
)

target_link_libraries(simple_radar_pipeline
  PRIVATE holoscan::core matx::matx
)

# Copy config file to the build tree
add_custom_target(simple_radar_pipeline_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/simple_radar_pipeline.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "simple_radar_pipeline.yaml"
    BYPRODUCTS "simple_radar_pipeline.yaml"
)
add_dependencies(simple_radar_pipeline simple_radar_pipeline_yaml)

# Add testing
if(BUILD_TESTING)
  add_test(NAME simple_radar_pipeline_test
           COMMAND simple_radar_pipeline
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(simple_radar_pipeline_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Scheduler finished")
endif()
