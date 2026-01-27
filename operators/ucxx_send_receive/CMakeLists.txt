# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)

project(ucxx_send_receive_operators LANGUAGES CXX)

find_package(holoscan 3.8 REQUIRED CONFIG)

add_library(ucxx_common STATIC
  ucxx_endpoint.cpp
  serialize_tensor.cpp
)
add_library(holoscan::ops::ucxx_common ALIAS ucxx_common)

target_include_directories(ucxx_common
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(ucxx_common
  PUBLIC
    holoscan::core
    ucs
    ucxx
)

add_subdirectory(receiver_op)
add_subdirectory(sender_op)

add_library(ucxx_send_receive INTERFACE)
target_link_libraries(ucxx_send_receive INTERFACE
  holoscan::ops::ucxx_sender_op
  holoscan::ops::ucxx_receiver_op
)
add_library(holoscan::ops::ucxx_send_receive ALIAS ucxx_send_receive)
add_library(holoscan::ops::ucxx ALIAS ucxx_send_receive)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
