# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(xr)
find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(Vulkan REQUIRED)
find_package(Eigen3 3.4 REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

enable_language(CUDA)

add_subdirectory(deps)

add_library(
  xr
  xr_begin_frame_op.cpp
  xr_composition_layers.cpp
  xr_end_frame_op.cpp
  xr_hand_tracker.cpp
  xr_composition_layer_manager.cpp
  xr_session.cpp
  xr_swapchain_cuda.cpp)

add_library(holoscan::ops::xr ALIAS xr)

target_compile_definitions(
  xr
  PUBLIC # Don't link vulkan function prototypes to ensure they are loaded dynamically.
         VK_NO_PROTOTYPES
         # Disable constructors to allow use of designated initializers.
         VULKAN_HPP_NO_CONSTRUCTORS
         # Ensure that all code uses the dynamic Vulkan-Hpp RAII dispatcher.
         VULKAN_HPP_NO_DEFAULT_DISPATCHER
         # Enable OpenXR Vulkan extensions.
         XR_USE_GRAPHICS_API_VULKAN)

target_compile_options(xr PUBLIC -Wno-terminate)

target_include_directories(xr INTERFACE ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR})

target_link_libraries(
  xr
  PUBLIC holoscan::core openxr_headers openxr_loader Vulkan::Headers Vulkan::Vulkan glm::glm
  PRIVATE holoscan::viz Eigen3::Eigen)

set_property(TARGET xr PROPERTY POSITION_INDEPENDENT_CODE ON)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
