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
cmake_minimum_required(VERSION 3.20)
project(volume_loader)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Set CMP0135 policy to NEW to use time of extraction for files extracted by
# FetchContent/ExternalProject_Add.
# This ensures that anything that depends on the extracted contents will be rebuilt whenever
# the URL changes.
if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
  set(CMAKE_POLICY_DEFAULT_CMP0135 NEW)
endif()

# NIFTI c lib
include(FetchContent)
FetchContent_Declare(
  nifti_clib
  GIT_REPOSITORY "https://github.com/NIFTI-Imaging/nifti_clib.git"
  GIT_TAG "v3.0.0"
)

set(NIFTI_INSTALL_NO_DOCS ON CACHE INTERNAL "")
set(NIFTI_BUILD_TESTING OFF CACHE INTERNAL "")
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)
FetchContent_MakeAvailable(nifti_clib)

add_library(volume_loader SHARED
  mhd_loader.cpp
  mhd_loader.hpp
  nifti_loader.cpp
  nifti_loader.hpp
  nrrd_loader.cpp
  nrrd_loader.hpp
  volume_loader.cpp
  volume_loader.hpp
  volume.cpp
  volume.hpp
  )

add_library(holoscan::ops::volume_loader ALIAS volume_loader)

target_link_libraries(volume_loader
  PRIVATE
    holoscan::core
    NIFTI::nifti2
  )

target_include_directories(volume_loader INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(volume_loader PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR})
target_compile_definitions(volume_loader PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR})

install(TARGETS volume_loader
  COMPONENT holoscan-ops
)
