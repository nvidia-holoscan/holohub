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

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

set(HOLOSCAN_CROSS_SYSROOT "/usr/aarch64-linux-gnu" CACHE PATH "GNU AArch64 sysroot")
set(HOLOSCAN_TARGET_SDK_ROOT "/opt/nvidia/holoscan" CACHE PATH "AArch64 Holoscan SDK prefix")

set(_HOLOSCAN_CROSS_NVCC "${CMAKE_CURRENT_LIST_DIR}/nvcc-cross-sbsa")

list(APPEND CMAKE_TRY_COMPILE_PLATFORM_VARIABLES
  HOLOSCAN_CROSS_SYSROOT
  HOLOSCAN_TARGET_SDK_ROOT)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

set(CMAKE_CUDA_COMPILER "${_HOLOSCAN_CROSS_NVCC}")
set(CMAKE_CUDA_HOST_COMPILER aarch64-linux-gnu-g++)
set(CUDAToolkit_ROOT /usr/local/cuda-13.0)
set(CUDAToolkit_NVCC_EXECUTABLE "${_HOLOSCAN_CROSS_NVCC}")

list(PREPEND CMAKE_FIND_ROOT_PATH
  "${HOLOSCAN_TARGET_SDK_ROOT}"
  "${HOLOSCAN_CROSS_SYSROOT}"
  "/usr/local/cuda-13.0/targets/sbsa-linux")
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

unset(_HOLOSCAN_CROSS_NVCC)
