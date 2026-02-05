# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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

# Set policy to use <PACKAGENAME>_ROOT variables in find_package
if(POLICY CMP0144)
  cmake_policy(SET CMP0144 NEW)
endif()

# Check if EC-Master SDK is available before building
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR})
find_package(EcMaster QUIET)

if(EcMaster_FOUND)
  message(STATUS "Building HoloCat application with EC-Master SDK")
  add_subdirectory(cpp)

  message(STATUS "Building HoloCat tests")
  add_subdirectory(tests)
else()
  message(
    STATUS
    "EC-Master SDK not found. Skipping HoloCat application build.")
endif()
