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
project(stereo_depth CXX CUDA)

find_package(holoscan 2.4 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


include(FetchContent)
FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(Eigen3)

add_executable(stereo_depth
  main.cpp
  undistort_rectify.cpp
  split_video.cpp
  heat_map.cpp
  stereo_depth_kernels.cu
  crop.cpp
  ess_processor.cpp
  tracking_postprocessor.cpp
)
target_link_libraries(stereo_depth
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::holoviz
  holoscan::ops::v4l2
  holoscan::ops::format_converter
  holoscan::ops::inference
  holoscan::ops::inference_processor
  CUDA::nppif
  CUDA::nppidei
  CUDA::nppicc
  CUDA::nppial
  Eigen3::Eigen
)

# Copy config file
add_custom_target(stereo_depth_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/stereo_vision.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "stereo_vision.yaml"
    BYPRODUCTS "stereo_vision.yaml"
)

add_custom_command(
    OUTPUT "${HOLOHUB_DATA_DIR}/stereo_vision/yolov8-nms-update.onnx"
           "${HOLOHUB_DATA_DIR}/stereo_vision/ess.engine"
    COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/../scripts/get_data_and_models.sh" "${HOLOHUB_DATA_DIR}/stereo_vision"
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/stereo_vision/ess.engine"
               "${HOLOHUB_DATA_DIR}/stereo_vision/yolov8-nms-update.onnx"
)

add_custom_target(get_data_and_models ALL
                  DEPENDS
                  "${HOLOHUB_DATA_DIR}/stereo_vision/ess.engine"
                  "${HOLOHUB_DATA_DIR}/stereo_vision/yolov8-nms-update.onnx")

add_dependencies(stereo_depth stereo_depth_yaml get_data_and_models)
