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
message(STATUS "=== PROCESSING streaming_client_demo_enhanced/cpp/CMakeLists.txt ===")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(holoscan REQUIRED)

# Add executable
add_executable(streaming_client_demo_enhanced
    streaming_client_demo.cpp
)

# Print success message
message(STATUS "✓ Executable target 'streaming_client_demo_enhanced' created")

# Set build directory
message(STATUS "Build streaming_client_demo_enhanced directory: ${CMAKE_BINARY_DIR}/streaming_client_demo_enhanced")

# Set compile definitions
target_compile_definitions(streaming_client_demo_enhanced
    PRIVATE
    HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
    HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
    HOLOSCAN_PATCH_VERSION=${holoscan_VERSION_PATCH}
)

# Set compile features
target_compile_features(streaming_client_demo_enhanced
    PUBLIC
    cxx_std_17
)

# Set compile options
target_compile_options(streaming_client_demo_enhanced
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-reorder>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
)

# Add include directories
target_include_directories(streaming_client_demo_enhanced 
    PUBLIC 
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(streaming_client_demo_enhanced
    PUBLIC
    holoscan::core
    holoscan::ops::holoviz
    holoscan::ops::video_stream_replayer
    holoscan::ops::v4l2
    holoscan::ops::format_converter
    holoscan::ops::streaming_client_enhanced
    holoscan::data_loggers::basic_console_logger
    ${OpenCV_LIBS}
    CUDA::cudart
    CUDA::cuda_driver
)

# Print success message
message(STATUS "✓ Libraries linked to streaming_client_demo_enhanced")

# Get target properties
get_target_property(TARGET_TYPE streaming_client_demo_enhanced TYPE)
message(STATUS "Target type: ${TARGET_TYPE}")

get_target_property(TARGET_SOURCES streaming_client_demo_enhanced SOURCES)
message(STATUS "Target sources: ${TARGET_SOURCES}")

message(STATUS "=== COMPLETED streaming_client_demo_enhanced/cpp/CMakeLists.txt ===")

# Copy libraries
add_custom_command(TARGET streaming_client_demo_enhanced PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib"
)

# Copy executable and config files
add_custom_command(TARGET streaming_client_demo_enhanced POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin"
)

# Copy YAML file if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml")
    add_custom_command(TARGET streaming_client_demo_enhanced POST_BUILD
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
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/streaming_client_demo.yaml"
    )
endif()

# Copy executable to bin directory
add_custom_command(TARGET streaming_client_demo_enhanced POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:streaming_client_demo_enhanced>" "${CMAKE_BINARY_DIR}/bin/"
)

# Install rules
install(
    TARGETS streaming_client_demo_enhanced
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Install config files
install(
    FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml"
    DESTINATION bin
)

# Copy config dependencies for testing
add_custom_target(streaming_client_demo_enhanced_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_client_demo.yaml"
  BYPRODUCTS "streaming_client_demo.yaml"
)
add_dependencies(streaming_client_demo_enhanced streaming_client_demo_enhanced_deps)


if(BUILD_TESTING)
  add_test(NAME streaming_client_demo_enhanced_test
           COMMAND streaming_client_demo_enhanced --help
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(streaming_client_demo_enhanced_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Usage: streaming_client_demo_enhanced"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed;error"
                       TIMEOUT 10)
  message(STATUS "Added application integration test: streaming_client_demo_enhanced_test")
endif()
