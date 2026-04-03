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

if(OP_hsb_roce_receiver_doca_gpunetio OR
   hsb_roce_receiver_doca_gpunetio IN_LIST HOLOHUB_BUILD_OPERATORS OR
   "\"hsb_roce_receiver_doca_gpunetio\"" IN_LIST HOLOHUB_BUILD_OPERATORS)
  message(STATUS "DOCA GPUNetIO support enabled for imx274_gpu_resident")
  target_compile_definitions(imx274_gpu_resident PRIVATE HOLOHUB_DOCA_GPUNETIO)
  target_link_libraries(imx274_gpu_resident
    holoscan::ops::hsb_roce_receiver_doca_gpunetio
  )
else()
  message(STATUS "DOCA GPUNetIO support disabled for imx274_gpu_resident")
endif()
