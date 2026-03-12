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

find_package(Hololink 2.5 REQUIRED CONFIG
    PATHS "/opt/nvidia/hololink")

# libibverbs from libibverbs-dev; use pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(IBVERBS REQUIRED libibverbs)

# RoCE receiver operator
add_library(hsb_roce_receiver_nmd SHARED
  roce_receiver_op.cpp
  roce_receiver_no_host_metadata.cpp
)

add_library(holoscan::ops::hsb_roce_receiver_nmd ALIAS hsb_roce_receiver_nmd)

target_include_directories(hsb_roce_receiver_nmd
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${IBVERBS_INCLUDE_DIRS}
)

target_link_libraries(hsb_roce_receiver_nmd
  PUBLIC
    hololink::core
    hololink::common
    hololink::operators::base_receiver
    hololink::operators::roce_receiver
    holoscan::core
    fmt::fmt
    CUDA::cuda_driver
    ${IBVERBS_LIBRARIES}
)

target_compile_definitions(hsb_roce_receiver_nmd
  PRIVATE HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
  PRIVATE HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
)

install(TARGETS hsb_roce_receiver_nmd
  COMPONENT holoscan-ops
)
