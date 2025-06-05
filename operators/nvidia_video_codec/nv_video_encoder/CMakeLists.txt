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

cmake_minimum_required(VERSION 3.24)
project(nv_video_encoder)

find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Detect system architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(LIB_ARCH_DIR "aarch64-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
    set(LIB_ARCH_DIR "x86_64-linux-gnu")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

find_library(NVCUVID_LIBRARY nvcuvid
    NAMES libnvcuvid.so.1 libnvcuvid
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    PATHS "/usr/lib/${LIB_ARCH_DIR}/")
 
 if(NOT NVCUVID_LIBRARY)
   message(FATAL_ERROR "nvcuvid library not found. Please specify its location manually.")
endif()

find_library(NVENC_LIBRARY nvidia-encode
    NAMES libnvidia-encode.so.1 libnvidia-encode
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    PATHS "/usr/lib/${LIB_ARCH_DIR}/")

if(NOT NVENC_LIBRARY)
    message(FATAL_ERROR "nvidia-encode library not found. Please specify its location manually.")
endif()


# publisher Operator
add_library(nv_video_encoder SHARED
  nv_video_encoder.cpp
  ${NVC_SDK_PATH}/NvEncoder/NvEncoder.cpp
  ${NVC_SDK_PATH}/NvEncoder/NvEncoderCuda.cpp
)

add_library(holoscan::ops::nv_video_encoder ALIAS nv_video_encoder)
target_link_libraries(nv_video_encoder PUBLIC
  holoscan::core
  cuda
  avcodec
  avformat
  avutil
  ${NVCUVID_LIBRARY}
  ${NVENC_LIBRARY}
)
target_include_directories(nv_video_encoder PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  ${NVC_SDK_PATH}/
  ${NVC_SDK_PATH}/NvEncoder
)

# Add third_party include directory
target_include_directories(nv_video_encoder
    PRIVATE
)

add_subdirectory(python)
