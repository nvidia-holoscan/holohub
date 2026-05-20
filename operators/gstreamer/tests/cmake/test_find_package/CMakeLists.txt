# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# CMake project-mode test: verifies find_package(holoscan-gstreamer) succeeds from
# the install tree.  holoscan-gstreamer_DIR is injected by the parent CTest
# invocation via -D; no @VAR@ substitution is needed.
#
# Must run in project mode (not script mode) because the installed targets file
# calls add_library(IMPORTED), which is not scriptable.

cmake_minimum_required(VERSION 3.20)
project(test_find_package LANGUAGES CXX)

find_package(holoscan-gstreamer REQUIRED NO_DEFAULT_PATH)

if(NOT TARGET holoscan::holoscan_gstreamer_bridge)
    message(FATAL_ERROR
        "find_package(holoscan-gstreamer) succeeded but target "
        "holoscan::holoscan_gstreamer_bridge was not defined. "
        "Check that the targets file was installed correctly.")
endif()

message(STATUS
    "PASS: find_package(holoscan-gstreamer) with "
    "holoscan-gstreamer_DIR=${holoscan-gstreamer_DIR} OK; "
    "target holoscan::holoscan_gstreamer_bridge is available")

# Build a trivial executable that includes a holoscan-gstreamer header and links
# against the imported target.  find_dependency(holoscan) was already called by
# the config file above, so all transitive targets (holoscan::core, yaml-cpp,
# CCCL, …) are defined here.  A separate cmake --build ctest step compiles and
# links this target, catching a wrong INSTALL_INTERFACE include path or a broken
# IMPORTED_LOCATION at build time rather than configure time.
add_executable(check_includes check_includes.cpp)
target_link_libraries(check_includes PRIVATE holoscan::holoscan_gstreamer_bridge)
