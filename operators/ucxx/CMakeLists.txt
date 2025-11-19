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

project(ucxx_ops LANGUAGES CXX)

add_library(ucxx_ops SHARED
  ucxx_endpoint.cc
  ucxx_receiver_op.cc
  ucxx_sender_op.cc
)
add_library(isaac::ops::ucxx ALIAS ucxx_ops)

target_include_directories(ucxx_ops
  PUBLIC ${CMAKE_SOURCE_DIR}
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}
)

include(isaac_deps/holoscan)
target_link_libraries(ucxx_ops
  PUBLIC
    flatbuffers::flatbuffers
    holoscan::core
    isaac::messages::reflection
    ucs
    ucxx
)

if(ISAAC_BUILD_TESTS)
  add_subdirectory(tests)
endif()
