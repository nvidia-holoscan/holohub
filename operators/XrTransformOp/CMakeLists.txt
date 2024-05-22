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

cmake_minimum_required(VERSION 3.20.1)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(Vulkan REQUIRED)
project(xr_transform_op)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

enable_language(CUDA)
# OpenXR-SDK-Source
FetchContent_Declare(
  openxr
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenXR-SDK-Source.git
  GIT_TAG dde23f1a37968b4b458d0de71b4f6b67ed16024a # release-1.0.26
  SOURCE_DIR OpenXR-SDK-Source
)
FetchContent_MakeAvailable(openxr)

# OpenXR-Hpp
FetchContent_Declare(
  openxr_hpp
  GIT_REPOSITORY https://github.com/KhronosGroup/OpenXR-Hpp.git
  GIT_TAG 5d8262f14476f1b0504a58bdd7e8b89eb36787db # release-1.0.26
  PATCH_COMMAND ./generate-openxr-hpp.sh || true
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/OpenXR-Hpp
  SOURCE_DIR OpenXR-Hpp
)
FetchContent_GetProperties(openxr_hpp)
if(NOT openxr_hpp_POPULATED)
  FetchContent_Populate(openxr_hpp)
  add_library(openxr_hpp INTERFACE)
  target_include_directories(openxr_hpp
    INTERFACE
    ${openxr_hpp_SOURCE_DIR}/include
  )
endif()

# Get nlohmann_json from ClaraViz
FetchContent_MakeAvailable(nlohmann_json)

add_library(xr_transform_op
            XrTransformControlOp/xr_transform_control_op.cpp
            XrTransformRenderOp/xr_transform_render_op.cpp
            ux/ux_bounding_box_controller.cpp
            ux/ux_bounding_box_renderer.cpp
            ux/ux_cursor.cpp
            ux/ux_window_controller.cpp
            ux/ux_window_renderer.cpp
)

add_library(holoscan::ops::xr_transform_op ALIAS xr_transform_op)

target_link_libraries(xr_transform_op
  PRIVATE
    CUDA::cudart
    GXF::core
    GXF::multimedia
    holoscan::core
    holoscan::viz
    holoscan::viz::imgui
  PUBLIC
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    openxr_loader
    openxr_hpp
    Vulkan::Vulkan
)

target_include_directories(xr_transform_op INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(xr_transform_op INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/XrTransformRenderOp)
target_include_directories(xr_transform_op INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/XrTransformControlOp)
target_include_directories(xr_transform_op INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/ux)

install(
  TARGETS xr_transform_op
  EXPORT holoscan-ops
)
