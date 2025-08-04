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

# Include the utility function for fetching Holohub operators
# This provides the fetch_holohub_operator() function that uses Git sparse checkout
# to efficiently download only the required operator from the Holohub repository
include(../../cmake/FetchHolohubOperator.cmake)

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
