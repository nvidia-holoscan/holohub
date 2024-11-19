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
  GIT_TAG v0.8.0
)
FetchContent_MakeAvailable(MatX)

add_executable(adv_networking_bench
  kernels.cu
  main.cpp
)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

target_link_libraries(adv_networking_bench
  PRIVATE
  holoscan::core
  holoscan::advanced_network_rx
  holoscan::advanced_network_tx
  matx::matx
)
# Initialize ANO_MGR with a default if not provided
if(NOT DEFINED ANO_MGR)
    set(ANO_MGR "dpdk doca" CACHE STRING "Manager type(s) list")
endif()

separate_arguments(ANO_MGR_LIST UNIX_COMMAND ${ANO_MGR})

# Define targets for each manager type
foreach(MGR IN LISTS ANO_MGR_LIST)
    message(STATUS "Adding manager ${MGR} to the configuration")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "ANO_MGR_${MGR}=1")
endforeach()


set_target_properties(adv_networking_bench PROPERTIES CUDA_ARCHITECTURES "80;90")

# Copy config file
add_custom_target(adv_networking_bench_default_tx_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml"
)

add_dependencies(adv_networking_bench adv_networking_bench_default_tx_rx_yaml)

add_custom_target(adv_networking_bench_default_tx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx.yaml"
)

add_dependencies(adv_networking_bench adv_networking_bench_default_tx_yaml)

add_custom_target(adv_networking_bench_default_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_rx.yaml"
)

add_dependencies(adv_networking_bench adv_networking_bench_default_rx_yaml)

add_custom_target(adv_networking_bench_doca_tx_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_doca_tx_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_doca_tx_rx.yaml"
)

add_dependencies(adv_networking_bench adv_networking_bench_doca_tx_rx_yaml)

add_custom_target(adv_networking_bench_rmax_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_rmax_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_rmax_rx.yaml"
)

add_dependencies(adv_networking_bench adv_networking_bench_rmax_rx_yaml)

# Installation
install(TARGETS adv_networking_bench
        DESTINATION bin/adv_networking_bench/cpp)

install(
  FILES ../adv_networking_bench_default_rx.yaml
        ../adv_networking_bench_default_tx.yaml
        ../adv_networking_bench_default_tx_rx.yaml
        ../adv_networking_bench_doca_tx_rx.yaml
        ../adv_networking_bench_rmax_rx_yaml
  DESTINATION bin/adv_networking_bench/cpp
)
