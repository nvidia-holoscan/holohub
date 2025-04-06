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

project(rivermax_service)

add_library(${PROJECT_NAME} INTERFACE)

target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
# To enable CUDA support for Rivermax, we need to set RMAX_CUDA to TRUE
# The parameter is passed to the RDK Build system
set(RMAX_CUDA TRUE)

# Check if the environment variable RMAX_TEGRA is set
# RMAX_TEGRA controls whether the build targets NVIDIA's Tegra platform (default is OFF).
# The parameter is passed to the RDK Build system
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
    target_compile_definitions(${PROJECT_NAME} INTERFACE RMAX_TEGRA=1)
else()
    target_compile_definitions(${PROJECT_NAME} INTERFACE RMAX_TEGRA=0)
endif()

# Clear any previously cached value
unset(RIVERMAX_DEV_KIT_REPO_URL CACHE)

# Configure FetchContent for Rivermax library
include(FetchContent)

# Force position-independent code for all targets to avoid linker issues
set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# For fine-grained tokens, use the token directly in the URL
set(RIVERMAX_DEV_KIT_REPO_URL "https://github.com/NVIDIA/rivermax-dev-kit.git" CACHE STRING "Repository URL" FORCE)

# Clear and set with FORCE to prevent overrides
unset(RIVERMAX_DEV_KIT_TAG CACHE)
set(RIVERMAX_DEV_KIT_TAG "ef340f769869d75080f63f91a0a1d7e36ccc4688" CACHE STRING "Git tag, branch or branch hash to use for rivermax-dev-kit" FORCE)
message(STATUS "Using Rivermax Hash: ${RIVERMAX_DEV_KIT_TAG}")

FetchContent_Declare(
    rivermax-dev-kit
    GIT_REPOSITORY ${RIVERMAX_DEV_KIT_REPO_URL}
    GIT_TAG ${RIVERMAX_DEV_KIT_TAG}
    GIT_PROGRESS TRUE
)

FetchContent_MakeAvailable(rivermax-dev-kit)

# Link external RDK libraries through INTERFACE
target_link_libraries(${PROJECT_NAME} INTERFACE
    rivermax-dev-kit-app-base
    rivermax-dev-kit
)

# Add object libraries to get the actual implementations
target_sources(${PROJECT_NAME} INTERFACE
    $<TARGET_OBJECTS:rmax_rtp_receiver_obj>
    $<TARGET_OBJECTS:rmax_ipo_receiver_obj>
    $<TARGET_OBJECTS:rmax_xstream_media_sender_obj>
)
