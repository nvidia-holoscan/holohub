# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(frame_skipper frame_skipper.cpp)
add_library(holoscan::frame_skipper ALIAS frame_skipper)

target_compile_features(frame_skipper PUBLIC cxx_std_20)
target_include_directories(frame_skipper
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(frame_skipper PUBLIC holoscan::v4l2_depth_common)
