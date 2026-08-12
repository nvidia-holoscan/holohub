# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(v4l2_depth_common INTERFACE)
add_library(holoscan::v4l2_depth_common ALIAS v4l2_depth_common)

target_compile_features(v4l2_depth_common INTERFACE cxx_std_20)
target_include_directories(v4l2_depth_common
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(v4l2_depth_common INTERFACE holoscan::core)
