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
project(streaming_server_demo CXX)

# Add executable
add_executable(streaming_server_demo
    streaming_server_demo.cpp
)

# Print success message
message(STATUS "✓ Executable target 'streaming_server_demo' created")

# Set build directory
message(STATUS "Build streaming_server_demo directory: ${CMAKE_BINARY_DIR}/streaming_server_demo")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(holoscan REQUIRED)

# Set compile definitions
target_compile_definitions(streaming_server_demo
    PRIVATE
    HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
    HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
    HOLOSCAN_PATCH_VERSION=${holoscan_VERSION_PATCH}
)

# Set compile options
target_compile_options(streaming_server_demo
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-reorder>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
)

# Link libraries
target_link_libraries(streaming_server_demo
    PUBLIC
    holoscan::core
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cuda_driver
)

# Print success message
message(STATUS "✓ Libraries linked to streaming_server_demo")

# Set target properties
set_target_properties(streaming_server_demo PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SKIP_BUILD_RPATH FALSE
)

# Add include directories
target_include_directories(streaming_server_demo
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Copy YAML file if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml")
    add_custom_command(TARGET streaming_server_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml"
        "$<TARGET_FILE_DIR:streaming_server_demo>/streaming_server_demo.yaml"
    )
endif()

# Create lib directory
add_custom_command(TARGET streaming_server_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:streaming_server_demo>/lib"
)

# Install rules
install(
    TARGETS streaming_server_demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Install config files
install(
    FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml"
    DESTINATION bin
)

