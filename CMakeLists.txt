# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.30.4)

project(holoscan_5_0_ea_samples
    VERSION 0.2.0
    DESCRIPTION "Reference samples for the Holoscan SDK 5.0 Early Access 2 API"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Threads REQUIRED)
find_package(holoscan 5.0.0 REQUIRED CONFIG)

# Keep discovery lightweight by default. The Holohub CLI enables the selected
# application and its declared operator dependencies explicitly.
option(BUILD_ALL "Build every sample operator and application" OFF)
option(HOLOSCAN_5_0_EA_SAMPLES_BUILD_TESTING "Build this project's tests" ON)

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(HoloHubConfigHelpers)

# Applications declare their operator dependencies before the operator
# discovery pass evaluates the corresponding OP_<name> options.
add_subdirectory(applications)
add_subdirectory(operators)

if(HOLOSCAN_5_0_EA_SAMPLES_BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()
