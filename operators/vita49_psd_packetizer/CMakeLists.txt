# SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(vita49_psd_packetizer CXX)

find_package(holoscan 2.5.0 REQUIRED CONFIG
  PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(matx CONFIG REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

add_subdirectory(packet_sender)
add_library(vita49_psd_packetizer STATIC
    vita49_psd_packetizer.cu
)
add_library(holoscan::ops::vita49_psd_packetizer ALIAS vita49_psd_packetizer)

target_link_libraries(vita49_psd_packetizer
    PUBLIC
    packet_sender
    PRIVATE
    matx::matx
    holoscan::core
)

target_include_directories(vita49_psd_packetizer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS vita49_psd_packetizer)
