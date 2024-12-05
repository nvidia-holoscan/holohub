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
project(gxf_emergent_source)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
# Create library
add_library(gxf_emergent_source_lib SHARED
  emergent_source.cpp
  emergent_source.hpp
)
target_link_libraries(gxf_emergent_source_lib
  PUBLIC
  CUDA::cudart
  CUDA::cuda_driver
  GXF::multimedia
  GXF::std
  holoscan::core
  yaml-cpp
  EmergentCamera
)
target_include_directories(gxf_emergent_source_lib
  SYSTEM
  BEFORE
  PUBLIC
  "/opt/EVT/eSDK/include"
)
target_link_directories(gxf_emergent_source_lib
  BEFORE
  PUBLIC
  "/opt/EVT/eSDK/lib/"
)

# Create extension
add_library(gxf_emergent_source SHARED
  emergent_ext.cpp
)
target_link_libraries(gxf_emergent_source
  PUBLIC gxf_emergent_source_lib
)
