# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_library(ucxx_send_receive SHARED
  ucxx_endpoint.cpp
  ucxx_receiver_op.cpp
  ucxx_sender_op.cpp
  serialize_tensor.cpp
)
add_library(holoscan::ops::ucxx ALIAS ucxx_send_receive)

target_include_directories(ucxx_send_receive
  PUBLIC $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}>
  PRIVATE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(ucxx_send_receive
  PUBLIC
    holoscan::core
    ucs
    ucxx
)

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
