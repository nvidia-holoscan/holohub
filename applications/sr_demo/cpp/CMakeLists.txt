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
project(sr_demo CXX CUDA)

find_package(holoscan 2.3 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


add_executable(sr_demo
  main.cpp
)
target_link_libraries(sr_demo
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::holoviz
  holoscan::ops::v4l2
  holoscan::ops::format_converter
  holoscan::ops::inference
  holoscan::ops::inference_processor
)

# Copy config file
add_custom_target(sr_demo_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/sr_demo.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "sr_demo.yaml"
    BYPRODUCTS "sr_demo.yaml"
)


add_dependencies(sr_demo sr_demo_yaml)
