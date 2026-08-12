# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(X11 REQUIRED)

add_library(x11_display x11_display.cpp)
add_library(holoscan::x11_display ALIAS x11_display)

target_compile_features(x11_display PUBLIC cxx_std_20)
target_include_directories(x11_display
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(x11_display
    PUBLIC
        holoscan::v4l2_depth_common
        X11::X11
)
