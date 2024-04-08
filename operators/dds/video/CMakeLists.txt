# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.24)
project(dds_video_operators)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Publisher Operator
option("OP_dds_video_publisher" "Build the DDS Video Publisher operator" ${BUILD_ALL})
if(${OP_dds_video_publisher})
  add_library(dds_video_publisher SHARED publisher.cpp publisher.hpp)
  add_library(holoscan::ops::dds_video_publisher ALIAS dds_video_publisher)
  target_link_libraries(dds_video_publisher PUBLIC
    dds_operator_base
    dds_video_frame
  )
  target_include_directories(dds_video_publisher PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  set("LIB_dds_operator_base" ON CACHE BOOL "Build LIB_dds_operator_base" FORCE)
  set("LIB_dds_video_frame" ON CACHE BOOL "Build LIB_dds_video_frame" FORCE)
endif()

# Subscriber Operator
option("OP_dds_video_subscriber" "Build the DDS Video Subscriber operator" ${BUILD_ALL})
if(${OP_dds_video_subscriber})
  add_library(dds_video_subscriber SHARED subscriber.cpp subscriber.hpp)
  add_library(holoscan::ops::dds_video_subscriber ALIAS dds_video_subscriber)
  target_link_libraries(dds_video_subscriber PUBLIC
    dds_operator_base
    dds_video_frame
  )
  target_include_directories(dds_video_subscriber PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../..)
  set("LIB_dds_operator_base" ON CACHE BOOL "Build LIB_dds_operator_base" FORCE)
  set("LIB_dds_video_frame" ON CACHE BOOL "Build LIB_dds_video_frame" FORCE)
endif()

# DDS generated type library
option("LIB_dds_video_frame" "Build the DDS Video Frame library" ${BUILD_ALL})
if(${LIB_dds_video_frame})
  include(RTIConnextDDS)
  add_rti_type_library(dds_video_frame ${CMAKE_CURRENT_SOURCE_DIR}/VideoFrame.idl UNBOUNDED)
endif()

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
