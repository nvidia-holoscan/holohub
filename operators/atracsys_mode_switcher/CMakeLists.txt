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
project(atracsys_mode_switcher LANGUAGES CXX)

find_package(holoscan 4.0.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)

add_library(atracsys_mode_switcher SHARED
  mode_switcher_keyboard.cpp
  mode_switcher_op.cpp
)
add_library(holoscan::ops::atracsys_mode_switcher ALIAS atracsys_mode_switcher)

target_include_directories(atracsys_mode_switcher
  PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(atracsys_mode_switcher
  PUBLIC
    holoscan::core
    CUDA::cudart
)

install(TARGETS atracsys_mode_switcher)
