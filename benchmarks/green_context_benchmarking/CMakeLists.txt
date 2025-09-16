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
project(green_context_benchmarking CXX CUDA)

find_package(holoscan 3.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find CUDA and CUPTI
find_package(CUDAToolkit REQUIRED)

# Enable CUDA language
enable_language(CUDA)

add_executable(green_context_benchmarking
    green_context_benchmark.cpp
    benchmark_cuda_kernel.cu
)

target_link_libraries(green_context_benchmarking
    PRIVATE
    holoscan::core
    CUDA::cuda_driver
    CUDA::cudart
    CUDA::cupti
)

# Set CUDA properties
set_property(TARGET green_context_benchmarking PROPERTY CUDA_STANDARD 17)
set_property(TARGET green_context_benchmarking PROPERTY CUDA_STANDARD_REQUIRED ON)
set_property(TARGET green_context_benchmarking PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Include CUDA headers
target_include_directories(green_context_benchmarking PRIVATE ${CUDAToolkit_INCLUDE_DIRS})