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
project(azure_kinect_camera)

find_package(holoscan 2.1 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(k4a REQUIRED CONFIG PATHS "/opt/tcn/3rdparty/direct_deploy")

add_library(azure_kinect_camera SHARED
  azure_kinect_camera.hpp
  azure_kinect_camera.cpp
)
add_library(holoscan::ops::azure_kinect_camera ALIAS azure_kinect_camera)

target_include_directories(
  azure_kinect_camera
  INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(
  azure_kinect_camera
  holoscan::core
  k4a::k4a
)

target_include_directories(azure_kinect_camera INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(azure_kinect_camera PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR})
target_compile_definitions(azure_kinect_camera PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR})

install(TARGETS azure_kinect_camera
        COMPONENT holoscan-ops
)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
