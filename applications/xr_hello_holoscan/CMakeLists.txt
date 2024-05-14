# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

include(FetchContent)
cmake_minimum_required(VERSION 3.20)
project(xr_hello_holoscan)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

FetchContent_Declare(
  Eigen3
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_MakeAvailable(Eigen3)

find_package(holoscan 2.0 REQUIRED CONFIG)

add_executable(xr_hello_holoscan main.cpp)
target_link_libraries(xr_hello_holoscan
  holoscan::core
  holoscan::ops::frame_op
  holoscan::ops::xr_transform_op
  holoscan::ops::xr_basic_render_op
  holoscan::viz
)
