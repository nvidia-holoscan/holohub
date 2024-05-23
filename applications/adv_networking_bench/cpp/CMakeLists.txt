# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(adv_networking_bench CXX CUDA)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(PkgConfig REQUIRED)

include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG cdcdffd
)
FetchContent_MakeAvailable(MatX)

add_executable(adv_networking_bench
  kernels.cu
  main.cpp
)

# set(CMAKE_BUILD_TYPE Debug)
# set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -ggdb")
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -ggdb")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G")

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

target_link_libraries(adv_networking_bench
  PRIVATE
  holoscan::core
  holoscan::advanced_network_rx
  holoscan::advanced_network_tx
  matx::matx
)

set_target_properties(adv_networking_bench PROPERTIES CUDA_ARCHITECTURES "80;90")

# Copy config file
add_custom_target(adv_networking_bench_default_tx_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml"
)

add_dependencies(adv_networking_bench adv_networking_bench_default_tx_rx_yaml)

add_custom_target(adv_networking_bench_doca_tx_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_doca_tx_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_doca_tx_rx.yaml"
)

add_dependencies(adv_networking_bench adv_networking_bench_doca_tx_rx_yaml)

