# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
cmake_minimum_required(VERSION 3.24)
project(simple_pdw_pipeline C CXX)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
list(PREPEND CMAKE_PREFIX_PATH "$ENV{DAQIRI_ROOT}" "/opt/daqiri")
find_package(daqiri REQUIRED CONFIG)
include(HoloHubDaqiri)

# Enable cuda language
enable_language(CUDA)

add_executable(simple_pdw_pipeline
  simple_pdw_pipeline.cu
)


target_link_libraries(simple_pdw_pipeline
  PRIVATE holoscan::core holoscan::matx
)
holohub_link_daqiri(simple_pdw_pipeline)

add_executable(simple_pdw_siggen
  siggen.c
)

target_link_libraries(simple_pdw_siggen
  pthread
)

# Copy config file to the build tree
add_custom_target(simple_pdw_pipeline_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${CMAKE_CURRENT_SOURCE_DIR}/simple_pdw_pipeline.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "simple_pdw_pipeline.yaml"
    BYPRODUCTS "simple_pdw_pipeline.yaml"
)
add_dependencies(simple_pdw_pipeline simple_pdw_pipeline_yaml simple_pdw_siggen)
