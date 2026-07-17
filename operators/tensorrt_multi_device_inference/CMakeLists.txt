# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(tensorrt_multi_device_inference LANGUAGES CXX CUDA)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)

# TensorRT >= 11 (Multi-Device GA) is required. find_library has no version
# constraint here; the >= 11 requirement is documented and enforced at runtime.
find_library(NVINFER_LIBRARY nvinfer REQUIRED)
find_library(NVINFER_PLUGIN_LIBRARY nvinfer_plugin REQUIRED)
find_library(NCCL_LIBRARY nccl REQUIRED)

add_library(tensorrt_multi_device_inference SHARED
  tensorrt_multi_device_inference.cpp
  tensorrt_multi_device_inference.hpp
  multidevice.cpp
  multidevice.hpp
)

# The reused MultiDeviceTrt core is gated behind this define in its source.
target_compile_definitions(tensorrt_multi_device_inference PRIVATE HOLOINFER_ENABLE_TRT_MULTI_DEVICE)

target_link_libraries(tensorrt_multi_device_inference
  PUBLIC
    holoscan::core
  PRIVATE
    ${NVINFER_LIBRARY}
    ${NVINFER_PLUGIN_LIBRARY}
    ${NCCL_LIBRARY}
    CUDA::cudart
)
target_include_directories(tensorrt_multi_device_inference INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
set_property(TARGET tensorrt_multi_device_inference PROPERTY CUDA_ARCHITECTURES "native")

install(TARGETS tensorrt_multi_device_inference)
