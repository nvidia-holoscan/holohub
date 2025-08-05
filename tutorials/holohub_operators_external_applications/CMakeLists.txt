# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Set minimum CMake version required for this project
# Version 3.18 is needed for modern CMake features and FetchContent
cmake_minimum_required(VERSION 3.18)
project(import_holohub_operators)

# Find and configure the Holoscan SDK
# This makes Holoscan targets available for linking (e.g., holoscan::core)
# The REQUIRED keyword will cause CMake to fail if Holoscan is not found
find_package(holoscan REQUIRED)

# =============================================================================
# FETCH HOLOHUB OPERATOR UTILITY - TWO APPROACHES AVAILABLE
# =============================================================================
# Choose one of the following two approaches by uncommenting the desired option:

# INTERNAL ONLY: Repository Include (for Holohub repository internal use only)
# This approach is only available when the application is within the Holohub repository structure
# For external applications, use one of the two approaches below
include(../../cmake/FetchHolohubOperator.cmake)

# APPROACH 1: Download from GitHub (for external applications with internet access)
# Pros: Independent of repository structure, automatic updates, no manual copying
# Cons: Requires internet connection, depends on GitHub availability
# Uncomment the following lines to use this approach (and comment out the internal include above):
# set(FETCH_HOLOHUB_OPERATOR_URL "https://raw.githubusercontent.com/nvidia-holoscan/holohub/refs/heads/main/cmake/FetchHolohubOperator.cmake")
# set(FETCH_HOLOHUB_OPERATOR_LOCAL_PATH "${CMAKE_CURRENT_BINARY_DIR}/FetchHolohubOperator.cmake")
# if(NOT EXISTS ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})
#      file(DOWNLOAD
#          ${FETCH_HOLOHUB_OPERATOR_URL}
#          ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH}
#          SHOW_PROGRESS
#          TLS_VERIFY ON
#      )
#      if(NOT EXISTS ${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})
#          message(FATAL_ERROR "Failed to download FetchHolohubOperator.cmake from ${FETCH_HOLOHUB_OPERATOR_URL}")
#      endif()
# endif()
# include(${FETCH_HOLOHUB_OPERATOR_LOCAL_PATH})

# APPROACH 2: Local Copy (for offline environments or version control)
# Pros: Complete independence, no internet dependency, full version control
# Cons: Requires manual file copying, need to manually update
# Uncomment the following line to use this approach (requires FetchHolohubOperator.cmake in same directory):
# include(${CMAKE_CURRENT_SOURCE_DIR}/FetchHolohubOperator.cmake)

# Fetch the AJA source operator from Holohub
# This will download the aja_source operator using Git sparse checkout
# The operator will be available for linking as holoscan::aja
fetch_holohub_operator(aja_source)

# Create the executable from the main.cpp source file
add_executable(${PROJECT_NAME} main.cpp)

# Link the executable against the required Holohub libraries
# PRIVATE means these dependencies are not propagated to other targets
target_link_libraries(${PROJECT_NAME}
   PRIVATE
   holoscan::core    # Core Holoscan functionality (Application, Operator base classes)
   holoscan::aja     # AJA video capture operator from Holohub
   )
