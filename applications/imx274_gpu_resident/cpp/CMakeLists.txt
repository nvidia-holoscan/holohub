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

find_package(holoscan 4.0.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Hololink's 'H' must be capitalized for FindPackageModeule to work
find_package(Hololink 2.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/hololink")


enable_language(CUDA)

add_executable(imx274_gpu_resident
  imx274_gpu_resident.cpp
  data_ready_input_op.cpp
  data_ready_kernels.cu
  copy_frame_kernels.cu
)

set_target_properties(imx274_gpu_resident PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_link_libraries(imx274_gpu_resident
  holoscan::core
  holoscan::ops::hsb_roce_receiver_nmd
  holoscan::ops::csi_to_bayer_gpu_resident
  csi_to_bayer_converter_base
  holoscan::ops::image_processor_gpu_resident
  holoscan::ops::display_gpu_resident
  holoscan::ops::bayer_demosaic_gpu_resident
  hololink::core
  hololink::common
  hololink::sensors::native_imx274_camera_sensor
  fmt::fmt
  CUDA::cuda_driver
)

# Optional DOCA GPUNetIO support.
# Enable explicitly with the 'doca' mode or:
#   --configure-args='-DHOLOHUB_ENABLE_DOCA_GPUNETIO=ON'
option(HOLOHUB_ENABLE_DOCA_GPUNETIO
       "Enable DOCA GPUNetIO support for imx274_gpu_resident"
       OFF)

if(HOLOHUB_ENABLE_DOCA_GPUNETIO)
  set(DOCA_PATH "/opt/mellanox/doca")

  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(DOCA_LIB_DIR "${DOCA_PATH}/lib/x86_64-linux-gnu")
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(DOCA_LIB_DIR "${DOCA_PATH}/lib/aarch64-linux-gnu")
  else()
    set(DOCA_LIB_DIR "${DOCA_PATH}/lib")
  endif()

  find_path(DOCA_INCLUDE_DIR doca_verbs.h
    PATHS ${DOCA_PATH}/include
    NO_DEFAULT_PATH
  )
  find_path(DOCA_DEV_INCLUDE_DIR doca_dev.h
    PATHS ${DOCA_PATH}/include
    NO_DEFAULT_PATH
  )
  find_path(DOCA_GPUNETIO_INCLUDE_DIR doca_gpunetio.h
    PATHS ${DOCA_PATH}/include
    NO_DEFAULT_PATH
  )
  find_library(DOCA_VERBS_LIB doca_verbs
    PATHS ${DOCA_LIB_DIR}
    NO_DEFAULT_PATH
  )
  find_library(DOCA_GPUNETIO_LIB doca_gpunetio
    PATHS ${DOCA_LIB_DIR}
    NO_DEFAULT_PATH
  )
  find_library(DOCA_COMMON_LIB doca_common
    PATHS ${DOCA_LIB_DIR}
    NO_DEFAULT_PATH
  )

  if(DOCA_INCLUDE_DIR AND DOCA_DEV_INCLUDE_DIR AND DOCA_GPUNETIO_INCLUDE_DIR
     AND DOCA_VERBS_LIB AND DOCA_COMMON_LIB AND DOCA_GPUNETIO_LIB)
    message(STATUS "DOCA GPUNetIO support enabled for imx274_gpu_resident "
                   "(found at ${DOCA_PATH})")
    target_compile_definitions(imx274_gpu_resident PRIVATE HOLOHUB_DOCA_GPUNETIO)
    target_link_libraries(imx274_gpu_resident
      holoscan::ops::hsb_roce_receiver_doca_gpunetio
    )
  else()
    message(FATAL_ERROR "DOCA SDK not found at ${DOCA_PATH} -- cannot build "
                       "with DOCA GPUNetIO support")
  endif()
else()
  message(STATUS "DOCA GPUNetIO support disabled for imx274_gpu_resident")
endif()
