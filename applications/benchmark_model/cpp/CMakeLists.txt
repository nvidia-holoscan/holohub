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
project(benchmark_model CXX)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(nvcv_types REQUIRED)
find_package(cvcuda REQUIRED)
find_package(OpenCV REQUIRED)

add_executable(benchmark_model
    cvcuda_utils.cpp
    gxf_utils.cpp
    benchmark_model.cpp
)

target_link_libraries(benchmark_model
    PRIVATE
    holoscan::core
    holoscan::ops::v4l2
    holoscan::ops::format_converter
    holoscan::ops::holoviz
    holoscan::ops::inference
    holoscan::ops::segmentation_postprocessor
    nvcv_types
    cvcuda
)

add_executable(benchmark_image_model
    benchmark_image_model.cpp
    cvcuda_utils.cpp
    gxf_utils.cpp
)

# set the path to the parent of libnvcv_samples_common.so to be NVCV_SAMPLES_COMMON
get_filename_component(NVCV_SAMPLES_COMMON ${NVCV_SAMPLES_COMMON_LIB} DIRECTORY)

target_link_libraries(benchmark_image_model
    PRIVATE
    holoscan::core
    holoscan::ops::format_converter
    holoscan::ops::holoviz
    holoscan::ops::inference
    nvcv_types
    cvcuda
    CUDA::nvjpeg
    nvcv_samples_common
    ${OpenCV_LIBS}
)

# Copy config file
add_custom_target(benchmark_model_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_model.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "benchmark_model.yaml"
    BYPRODUCTS "benchmark_model.yaml"
)
add_dependencies(benchmark_model benchmark_model_yaml)
add_dependencies(benchmark_image_model benchmark_model_yaml)
