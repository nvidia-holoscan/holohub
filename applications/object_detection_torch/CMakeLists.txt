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
project(object_detection_torch CXX)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(object_detection_torch
    main.cpp
)

target_link_libraries(object_detection_torch
    PRIVATE
    holoscan::core
    holoscan::ops::aja
    holoscan::ops::video_stream_replayer
    holoscan::ops::format_converter
    holoscan::ops::inference
    holoscan::ops::inference_processor
    holoscan::ops::holoviz
)

# Copy config file
add_custom_target(object_detection_torch_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/object_detection_torch.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "object_detection_torch.yaml"
    BYPRODUCTS "object_detection_torch.yaml"
)
add_dependencies(object_detection_torch object_detection_torch_yaml)
