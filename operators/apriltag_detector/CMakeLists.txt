# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# we don't expose the C++ interface of the operator yet, therefore link
# statically
cmake_minimum_required(VERSION 3.20)
project(apriltag_detector LANGUAGES CXX)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(apriltag_detector SHARED
  apriltag_detector.hpp
  apriltag_detector.cpp
  )

add_library(holoscan::apriltag_detector ALIAS apriltag_detector)

target_include_directories(apriltag_detector
  INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}
  )

target_include_directories(apriltag_detector
  PRIVATE
    /opt/nvidia/cu-april-tags/include
  )

# Define a variable to store the directory path
set(APRILTAG_LIBRARY_DIR "/opt/nvidia/cu-april-tags/lib")

target_link_libraries(apriltag_detector
  PRIVATE
    holoscan::core
    CUDA::nvrtc
    CUDA::nppicc
    CUDA::cuda_driver
    ${APRILTAG_LIBRARY_DIR}/libcuapriltags.a
  )

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
