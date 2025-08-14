# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(streaming_server_demo)

# Add C++ and Python subdirectories

# Add C++ subdirectory
add_subdirectory(cpp)

# Add Python subdirectory
add_subdirectory(python)

# Copy Python files to build directory
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/python/streaming_server_demo.py
    ${CMAKE_BINARY_DIR}/streaming_server_demo.py
    COPYONLY
)