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

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

set(REL_PTH_MATLAB_CODEGEN "codegen")  # Needs changing if building on Jetson

set(EXECUTABLES matlab_beamform)

add_executable(matlab_beamform
  main.cpp
)

add_library(matlab_beamform_lib SHARED IMPORTED)
set_target_properties(matlab_beamform_lib
  PROPERTIES IMPORTED_LOCATION
  "${CMAKE_CURRENT_SOURCE_DIR}/${REL_PTH_MATLAB_CODEGEN}/dll/matlab_beamform/matlab_beamform.so"
)

target_include_directories(matlab_beamform
  PRIVATE
  "${CMAKE_CURRENT_SOURCE_DIR}/${REL_PTH_MATLAB_CODEGEN}/dll/matlab_beamform"
  "$ENV{MATLAB_ROOT}/extern/include"
)

target_link_libraries(matlab_beamform
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  matlab_beamform_lib
  matlab_utils
)

target_include_directories(matlab_beamform
    PRIVATE
    ${PROJECT_SOURCE_DIR}/matlab_utils
)

add_custom_target("matlab_beamform_deps"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/matlab_beamform.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "matlab_beamform.yaml"
  BYPRODUCTS "matlab_beamform.yaml"
)
add_dependencies(matlab_beamform "matlab_beamform_deps")