# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(CUDAToolkit REQUIRED)

add_library(v4l2_source v4l2_source.cpp)
add_library(holoscan::v4l2_source ALIAS v4l2_source)

target_compile_features(v4l2_source PUBLIC cxx_std_20)
target_include_directories(v4l2_source
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(v4l2_source
    PUBLIC holoscan::v4l2_depth_common
    PRIVATE CUDA::cudart
)
