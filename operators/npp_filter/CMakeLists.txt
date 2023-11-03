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
project(npp_filter)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(npp_filter SHARED
  npp_filter.cpp
  npp_filter.hpp
  )

add_library(holoscan::ops::npp_filter ALIAS npp_filter)

target_link_libraries(npp_filter
  PRIVATE
    holoscan::core
    CUDA::nppif
    GXF::multimedia
  )

target_include_directories(npp_filter INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_definitions(npp_filter PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR})
target_compile_definitions(npp_filter PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR})
