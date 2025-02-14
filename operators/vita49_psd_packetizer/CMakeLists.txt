# SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(vita49_psd_packetizer CXX)

find_package(holoscan 2.5.0 REQUIRED CONFIG
  PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

include(FetchContent)
FetchContent_Declare(
    vrtgen
    GIT_REPOSITORY https://github.com/Geontech/vrtgen.git
    GIT_TAG v0.7.14
)
FetchContent_MakeAvailable(vrtgen)

include(FetchContent)
FetchContent_Declare(
    vita49_psd
    GIT_REPOSITORY https://github.com/vts-i/vita49-psd.git
    GIT_TAG v0.1.0
)
FetchContent_MakeAvailable(vita49_psd)

FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG v0.9.0
)
FetchContent_MakeAvailable(MatX)

add_library(packetlib
    packet_sender.cpp
)
target_include_directories(packetlib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(packetlib PRIVATE
    holoscan::core
    vrtgen::vrtgen
    vita49_psd::vita49_psd
)

add_library(vita49_psd_packetizer
    vita49_psd_packetizer.cu
)
add_library(holoscan::ops::vita49_psd_packetizer ALIAS vita49_psd_packetizer)

target_link_libraries(vita49_psd_packetizer PRIVATE
    matx::matx
    holoscan::core
    packetlib
)

target_include_directories(vita49_psd_packetizer INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

install(TARGETS vita49_psd_packetizer)
