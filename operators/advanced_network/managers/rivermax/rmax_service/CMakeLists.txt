# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
cmake_minimum_required(VERSION 3.20)

project(rmax_service)

add_library(${PROJECT_NAME} SHARED
  rmax_base_service.cpp
  rmax_ipo_receiver_service.cpp
  ipo_receiver_io_node.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

#get_target_property(CORE_COMPILE_FEATURES holoscan::core INTERFACE_COMPILE_FEATURES)
#message("Holoscan Core compile features: ${CORE_COMPILE_FEATURES}")
#target_compile_features(${PROJECT_NAME} PUBLIC ${CORE_COMPILE_FEATURES})


target_compile_features(${PROJECT_NAME}
PUBLIC cxx_std_17
)
target_compile_definitions(${PROJECT_NAME} PUBLIC RMAX_APPS_LIB_FLAT_STRUCTURE)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)


# add rmax_apps_lib with absolute path as a library
# Initialize ANO_MGR with a default if not provided
if(NOT DEFINED RAL_DIR_PATH)
    set(RAL_DIR_PATH "/opt/rivermax_sdk/rmax_apps_lib" CACHE STRING "Rmax Apps Lib absolute path")
endif()

set(RMAX_CUDA TRUE)

# Check if the environment variable RMAX_TEGRA is set
if(NOT DEFINED ENV{RMAX_TEGRA})
    message(STATUS "Environment variable RMAX_TEGRA was not defined. Defaulting to FALSE")
    set(RMAX_TEGRA FALSE)
else()
    # Convert the environment variable to CMake boolean TRUE or FALSE
    if("$ENV{RMAX_TEGRA}" STREQUAL "TRUE" OR "$ENV{RMAX_TEGRA}" STREQUAL "ON")
        set(RMAX_TEGRA TRUE)
    else()
        set(RMAX_TEGRA FALSE)
    endif()
endif()

unset(RMAX_TEGRA CACHE)
message(STATUS "RMAX_TEGRA is set to: ${RMAX_TEGRA}")

if(RMAX_TEGRA)
    target_compile_definitions(${PROJECT_NAME} PUBLIC RMAX_TEGRA=1)
else()
    target_compile_definitions(${PROJECT_NAME} PUBLIC RMAX_TEGRA=0)
endif()

add_subdirectory(${RAL_DIR_PATH} ${CMAKE_BINARY_DIR}/external_build)


target_link_libraries(${PROJECT_NAME} PUBLIC rmax-ral-build rmax-ral-lib)

#find_package(fmt)

#if(fmt_FOUND)
#  target_link_libraries(${PROJECT_NAME} PUBLIC fmt::fmt)
#else()
#  message(FATAL_ERROR "fmt library not found")
#endif()
