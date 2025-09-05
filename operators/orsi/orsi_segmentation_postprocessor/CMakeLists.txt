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
project(orsi_segmentation_postprocessor)
project(${PROJECT_NAME} LANGUAGES CXX CUDA)


find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(${PROJECT_NAME}
  SHARED
  segmentation_postprocessor.hpp
  segmentation_postprocessor.cuh
  segmentation_postprocessor.cpp
  segmentation_postprocessor.cu
)

set_target_properties(${PROJECT_NAME}  PROPERTIES CUDA_ARCHITECTURES "80")

add_library(holoscan::orsi::segmentation_postprocessor ALIAS ${PROJECT_NAME})
target_include_directories(${PROJECT_NAME} INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME}
PUBLIC
    holoscan::core
PRIVATE
  CUDA::nppidei
  CUDA::nppig
  CUDA::nppicc
  holoscan::ops::segmentation_postprocessor
)


if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()
