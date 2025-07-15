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

# Detect system architecture if not already set
if(NOT DEFINED LIB_ARCH_DIR)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        set(LIB_ARCH_DIR "/usr/lib/aarch64-linux-gnu/nvidia")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64|AMD64")
        set(LIB_ARCH_DIR "/usr/lib/x86_64-linux-gnu")
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()

message(STATUS "Searching for NVIDIA Video Codec Libraries in ${LIB_ARCH_DIR}")
find_library(NVCUVID_LIBRARY nvcuvid
    NAMES libnvcuvid.so.1 libnvcuvid.so
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    PATHS "${LIB_ARCH_DIR}/")

if(NOT NVCUVID_LIBRARY)
    message(FATAL_ERROR "nvcuvid library not found. Please specify its location manually.")
endif()

find_library(NVENC_LIBRARY nvidia-encode
    NAMES libnvidia-encode.so.1 libnvidia-encode.so
    HINTS ${CUDA_TOOLKIT_ROOT_DIR}/lib64
    PATHS "${LIB_ARCH_DIR}/")

if(NOT NVENC_LIBRARY)
    message(FATAL_ERROR "nvidia-encode library not found. Please specify its location manually.")
endif()