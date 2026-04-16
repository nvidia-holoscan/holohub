# SPDX-FileCopyrightText: Copyright (c) 2026 Wayland Technologies. All rights reserved.
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
project(atracsys_camera LANGUAGES CXX CUDA)

if(NOT Atracsys_DIR AND DEFINED ENV{Atracsys_DIR})
  set(Atracsys_DIR "$ENV{Atracsys_DIR}")
endif()

if(NOT Atracsys_DIR AND DEFINED ATRACSYS_SDK_DIR)
  if(EXISTS "${ATRACSYS_SDK_DIR}/cmake")
    set(Atracsys_DIR "${ATRACSYS_SDK_DIR}/cmake")
  elseif(EXISTS "${ATRACSYS_SDK_DIR}/lib/cmake/Atracsys")
    set(Atracsys_DIR "${ATRACSYS_SDK_DIR}/lib/cmake/Atracsys")
  endif()
endif()

if(NOT Atracsys_DIR AND DEFINED ENV{ATRACSYS_SDK_DIR})
  if(EXISTS "$ENV{ATRACSYS_SDK_DIR}/cmake")
    set(Atracsys_DIR "$ENV{ATRACSYS_SDK_DIR}/cmake")
  elseif(EXISTS "$ENV{ATRACSYS_SDK_DIR}/lib/cmake/Atracsys")
    set(Atracsys_DIR "$ENV{ATRACSYS_SDK_DIR}/lib/cmake/Atracsys")
  endif()
endif()

if(NOT OpenCV_DIR AND DEFINED ENV{OpenCV_DIR})
  set(OpenCV_DIR "$ENV{OpenCV_DIR}")
endif()

if(NOT S3DK_ROOT AND DEFINED ENV{S3DK_ROOT})
  set(S3DK_ROOT "$ENV{S3DK_ROOT}")
endif()

find_package(holoscan 4.0.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)
find_package(Atracsys REQUIRED COMPONENTS SDK)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core calib3d imgproc highgui ximgproc)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/modules")
find_package(S3DK REQUIRED)

add_library(atracsys_camera SHARED
  sdk/helpers.cpp
  sdk/geometry_helper.cpp
  master_source_op.cpp
  point_cloud_filter_op.cpp
  point_cloud_filter.cu
)
add_library(holoscan::ops::atracsys_camera ALIAS atracsys_camera)

target_include_directories(atracsys_camera
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/sdk
    ${CMAKE_SOURCE_DIR}/operators/atracsys_mode_switcher
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(atracsys_camera
  PUBLIC
    holoscan::core
    CUDA::cudart
    Atracsys::SDK
    S3DK::S3DK
    TBB::tbb
    OpenMP::OpenMP_CXX
    ${OpenCV_LIBS}
)

target_compile_definitions(atracsys_camera PUBLIC WITH_CUDA=1)

install(TARGETS atracsys_camera)
