# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(tapnext_inference)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(tapnext_inference SHARED
  tapnext_inference.cpp
  tapnext_inference.hpp
  )

target_link_libraries(tapnext_inference holoscan::core)
target_include_directories(tapnext_inference INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()

# Installation
install(TARGETS tapnext_inference)

