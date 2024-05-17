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
project(dds_shapes_operators)

find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Build the dds operator base
set("BUILD_dds_operator_base" ON CACHE BOOL "Build dds_operator_base" FORCE)

# DDS generated type library
include(RTIConnextDDS)
add_rti_type_library(dds_shapetype ${RTI_CONNEXT_DDS_DIR}/resource/idl/ShapeType.idl)

# Subscriber Operator
add_library(dds_shapes_subscriber SHARED dds_shapes_subscriber.cpp dds_shapes_subscriber.hpp)
add_library(holoscan::ops::dds_shapes_subscriber ALIAS dds_shapes_subscriber)
target_link_libraries(dds_shapes_subscriber PUBLIC
  dds_operator_base
  dds_shapetype
)
target_include_directories(dds_shapes_subscriber PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)
