# SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights reserved.
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
project(gxf_qcap_source)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

enable_language(CUDA)

# Create library
add_library(gxf_qcap_source_lib SHARED
  qcap_queue.hpp
  qcap_source.cpp
  qcap_source.hpp
  images/no_device_png.o
  images/no_signal_png.o
  images/no_sdk_png.o
)

find_library(QCAP_LIBRARY
    NAMES qcap.l4t
    HINTS "/opt/yuan/qcap/lib"
)

find_path(QCAP_INCLUDE
    NAMES qcap.h
    HINTS "/opt/yuan/qcap/include"
)

if(QCAP_LIBRARY)
add_definitions(-DBUILD_WITH_QCAP_SDK)
target_link_libraries(gxf_qcap_source_lib
  PUBLIC
    ${QCAP_LIBRARY}
)
endif()

target_link_libraries(gxf_qcap_source_lib
  PUBLIC
    CUDA::cudart
    CUDA::cuda_driver
    CUDA::nppidei
    CUDA::nppicc
    GXF::multimedia
    GXF::std
    yaml-cpp
)

if(QCAP_INCLUDE)
target_include_directories(gxf_qcap_source_lib
  SYSTEM
  BEFORE
  PUBLIC
    ${QCAP_INCLUDE}
)
endif()

target_include_directories(gxf_qcap_source_lib
  PRIVATE
  stb
)

target_link_directories(gxf_qcap_source_lib
  BEFORE
  PUBLIC
  "/opt/yuan/qcap/lib/"
)

# Create extension
add_library(gxf_qcap_source SHARED
  qcap_ext.cpp
)

target_link_libraries(gxf_qcap_source
  PUBLIC gxf_qcap_source_lib
)
