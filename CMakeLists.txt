# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.20.1)
project(Holohub-internal CXX)

# set CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/deps)

# To configure which apps/operators to build
include(HoloHubConfigHelpers)

# Enable Testing
include(CTest)

# Options
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
option(HOLOHUB_BUILD_PYTHON "Build Holoscan SDK Python Bindings" ON)
option(BUILD_SAMPLE_APPS "Build Holoscan SDK sample applications" OFF)

# Set the default data directory
set(HOLOHUB_DATA_DIR "${CMAKE_BINARY_DIR}/data" CACHE PATH "Data Download directory")

# Build the extensions
add_subdirectory(gxf_extensions)

# Build the operators
add_subdirectory(operators)

# Build the applications
add_subdirectory(applications)

# Build the operators
add_subdirectory(operators)
