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
project(vpi_stereo CXX CUDA)

find_package(holoscan 2.4 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(vpi 3.2 REQUIRED)

include(FetchContent)
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(Eigen3)

add_executable(vpi_stereo
  main.cpp
  undistort_rectify.cpp
  split_video.cpp
  heat_map.cpp
  stereo_depth_kernels.cu
  crop.cpp
  vpi_stereo.cpp
)
target_link_libraries(vpi_stereo
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::holoviz
  holoscan::ops::v4l2
  holoscan::ops::format_converter
  CUDA::nppif
  CUDA::nppidei
  CUDA::nppicc
  CUDA::nppial
  Eigen3::Eigen
  vpi
)

# Copy configuration, calibration and source video files to output dir
add_custom_target(vpi_stereo_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/stereo_vision.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "stereo_vision.yaml"
    BYPRODUCTS "stereo_vision.yaml"
)

add_custom_target(vpi_stereo_calibration_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/stereo_calibration.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "stereo_calibration.yaml"
    BYPRODUCTS "stereo_calibration.yaml"
)

add_custom_target(vpi_stereo_mp4
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/stereo.mp4" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "stereo.mp4"
    BYPRODUCTS "stereo.mp4"
)

add_dependencies(vpi_stereo vpi_stereo_yaml vpi_stereo_calibration_yaml vpi_stereo_mp4)
