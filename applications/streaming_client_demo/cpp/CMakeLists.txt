# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Print status message
message(DEBUG "=== PROCESSING streaming_client_demo/cpp/CMakeLists.txt ===")

# Set source file
set(SOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.cpp")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(holoscan REQUIRED)

# Add executable
add_executable(streaming_client_demo
    streaming_client_demo.cpp
)

# Print success message
message(STATUS "✓ Executable target 'streaming_client_demo' created")

# Set build directory
message(STATUS "Build streaming_client_demo directory: ${CMAKE_BINARY_DIR}/streaming_client_demo")

# Set compile definitions
target_compile_definitions(streaming_client_demo
    PRIVATE
    HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
    HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
    HOLOSCAN_PATCH_VERSION=${holoscan_VERSION_PATCH}
)

# Set compile features
target_compile_features(streaming_client_demo
    PUBLIC
    cxx_std_17
)

# Set compile options
target_compile_options(streaming_client_demo
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-reorder>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
)

# Add include directories
target_include_directories(streaming_client_demo PUBLIC $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>)

# Link libraries
target_link_libraries(streaming_client_demo
    PUBLIC
    holoscan::core
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cuda_driver
)

# Print success message
message(STATUS "✓ Libraries linked to streaming_client_demo")

# Get target properties
get_target_property(TARGET_TYPE streaming_client_demo TYPE)
message(STATUS "Target type: ${TARGET_TYPE}")

get_target_property(TARGET_SOURCES streaming_client_demo SOURCES)
message(STATUS "Target sources: ${TARGET_SOURCES}")

message(DEBUG "=== COMPLETED streaming_client_demo/cpp/CMakeLists.txt ===")

# Set target properties
set_target_properties(streaming_client_demo PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SKIP_BUILD_RPATH FALSE
)

# Add include directories
target_include_directories(streaming_client_demo
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Copy libraries
add_custom_command(TARGET streaming_client_demo PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:streaming_client_demo>/lib"
)

# Copy executable and config files
add_custom_command(TARGET streaming_client_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin"
)

# Copy YAML file if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml")
    add_custom_command(TARGET streaming_client_demo POST_BUILD
        # Copy to current binary dir
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo.yaml"

        # Copy to bin directory
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "${CMAKE_BINARY_DIR}/bin/streaming_client_demo.yaml"

        # Copy to build root
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "${CMAKE_BINARY_DIR}/streaming_client_demo.yaml"

        # The build/streaming_client_demo directory
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/streaming_client_demo"
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "${CMAKE_BINARY_DIR}/streaming_client_demo/streaming_client_demo.yaml"

        # Copy to target directory
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "$<TARGET_FILE_DIR:streaming_client_demo>/streaming_client_demo.yaml"
    )
endif()

# Copy executable to bin directory
add_custom_command(TARGET streaming_client_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:streaming_client_demo>" "${CMAKE_BINARY_DIR}/bin/"
)

# Install rules
install(
    TARGETS streaming_client_demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Install config files
install(
    FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml"
    DESTINATION bin
)

# Copy to build directory
add_custom_command(TARGET streaming_client_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/streaming_client_demo"
)

# Copy to target directory
add_custom_command(TARGET streaming_client_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:streaming_client_demo>/lib"
)