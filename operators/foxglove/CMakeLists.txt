# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(foxglove)

find_package(holoscan 4.1 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)
include(FetchFoxgloveSdk)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

add_library(foxglove SHARED
  foxglove_publisher.cpp
  foxglove_publisher.hpp
)
add_library(holoscan::ops::foxglove ALIAS foxglove)

target_include_directories(foxglove
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/holoscan/foxglove>
)

target_link_libraries(foxglove
  PUBLIC
    holoscan::core
    foxglove::sdk
    CUDA::cudart
)

install(TARGETS foxglove)
install(FILES foxglove_publisher.hpp DESTINATION include/holoscan/foxglove)

if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
