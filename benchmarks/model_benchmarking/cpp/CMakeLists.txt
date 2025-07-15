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
project(model_benchmarking CXX)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(model_benchmarking
    model_benchmarking.cpp
)

target_link_libraries(model_benchmarking
    PRIVATE
    holoscan::core
    holoscan::ops::v4l2
    holoscan::ops::format_converter
    holoscan::ops::holoviz
    holoscan::ops::inference
    holoscan::ops::segmentation_postprocessor
    holoscan::ops::video_stream_replayer
)

# Copy config file
add_custom_target(benchmark_model_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/model_benchmarking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "model_benchmarking.yaml"
    BYPRODUCTS "model_benchmarking.yaml"
)
add_dependencies(model_benchmarking benchmark_model_yaml)
