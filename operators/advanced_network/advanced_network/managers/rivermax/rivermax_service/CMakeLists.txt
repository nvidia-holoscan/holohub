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

add_library(${PROJECT_NAME} SHARED
  #rmax_base_service.cpp
  #rmax_ipo_receiver_service.cpp
  #ipo_receiver_io_node.cpp
  # We'll add the rmax_ipo_receiver.cpp after fetching it
)

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_compile_features(${PROJECT_NAME}
  PUBLIC cxx_std_17
)
target_compile_definitions(${PROJECT_NAME} PUBLIC RMAX_APPS_LIB_FLAT_STRUCTURE)

set(CMAKE_POSITION_INDEPENDENT_CODE TRUE)

# Set RMAX_CUDA flag
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

# GitHub Access Configuration - Modified for fine-grained token
# Clear any previously cached value
unset(GITHUB_AUTH CACHE)
unset(RIVERMAX_DEV_KIT_REPO_URL CACHE)

# Get GitHub token from environment variable
if(DEFINED ENV{RIVERMAX_GITHUB_PAT})
    set(RIVERMAX_GITHUB_PAT $ENV{RIVERMAX_GITHUB_PAT} CACHE STRING "Rivermax GitHub PAT" FORCE)
else()
    message(FATAL_ERROR "RIVERMAX_GITHUB_PAT environment variable not set. Please set it with your GitHub Personal Access Token.")
endif()

# Configure FetchContent for Rivermax library
include(FetchContent)

# For fine-grained tokens, use the token directly in the URL
set(RIVERMAX_DEV_KIT_REPO_URL "https://ronyrad:${RIVERMAX_GITHUB_PAT}@github.com/ronyrad/rivermax-dev-kit-test.git" CACHE STRING "Repository URL" FORCE)
message(STATUS "Using HTTPS with fine-grained token authentication")

# Clear and set with FORCE to prevent overrides
unset(RIVERMAX_DEV_KIT_TAG CACHE)
set(RIVERMAX_DEV_KIT_TAG "T_RIVERMAX_DEV_KIT_TEST_0_0_5" CACHE STRING "Git tag, branch or commit hash to use for rivermax-dev-kit" FORCE)
message(STATUS "Using Rivermax tag: ${RIVERMAX_DEV_KIT_TAG}")

# Diagnostics to verify URL (masked token for security)
string(REGEX REPLACE "github_pat_[^@]*" "github_pat_***" MASKED_URL "${RIVERMAX_DEV_KIT_REPO_URL}")
message(STATUS "Repository URL (masked): ${MASKED_URL}")

FetchContent_Declare(
    rivermax_dev_kit
    GIT_REPOSITORY ${RIVERMAX_DEV_KIT_REPO_URL}
    GIT_TAG ${RIVERMAX_DEV_KIT_TAG}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
)

# Make the rivermax-dev-kit content available
FetchContent_GetProperties(rivermax_dev_kit)
if(NOT rivermax_dev_kit_POPULATED)
    message(STATUS "Fetching rivermax-dev-kit...")
    FetchContent_Populate(rivermax_dev_kit)

    # Add the fetched directory
    add_subdirectory(${rivermax_dev_kit_SOURCE_DIR} ${rivermax_dev_kit_BINARY_DIR} EXCLUDE_FROM_ALL)

    message(STATUS "rivermax-dev-kit fetched successfully to ${rivermax_dev_kit_SOURCE_DIR}")

    # Now add the rmax_ipo_receiver.cpp to our library
    target_sources(${PROJECT_NAME} PRIVATE
        ${rivermax_dev_kit_SOURCE_DIR}/source/apps/rmax_ipo_receiver/rmax_ipo_receiver.cpp
        ${rivermax_dev_kit_SOURCE_DIR}/source/apps/rmax_rtp_receiver/rmax_rtp_receiver.cpp
        ${rivermax_dev_kit_SOURCE_DIR}/source/apps/rmax_xstream_media_sender/rmax_xstream_media_sender.cpp
    )
endif()

# Explicitly include the rmax_ipo_receiver include directory
target_include_directories(${PROJECT_NAME} PUBLIC
    ${rivermax_dev_kit_SOURCE_DIR}/source/apps/rmax_ipo_receiver/include
    ${rivermax_dev_kit_SOURCE_DIR}/source/apps/rmax_rtp_receiver/include
    ${rivermax_dev_kit_SOURCE_DIR}/source/apps/rmax_xstream_media_sender/include
)

# Link to the fetched libraries
target_link_libraries(${PROJECT_NAME} PUBLIC
    rivermax-dev-kit-services
    rivermax-dev-kit-core
    rivermax-dev-kit-io-node
    rivermax-dev-kit
    rivermax-dev-kit-apps
)

# Installation
install(TARGETS ${PROJECT_NAME} COMPONENT advanced_network-rivermax)
