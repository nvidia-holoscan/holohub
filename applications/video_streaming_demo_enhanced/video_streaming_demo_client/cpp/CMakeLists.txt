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
find_package(holoscan REQUIRED)

# Bundle SSL libraries from streaming_client_enhanced operator
# Detect system architecture
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
  set(ARCH_DIR "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(ARCH_DIR "aarch64")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
set(CLIENT_OPERATOR_SSL_DIR "${CMAKE_SOURCE_DIR}/operators/video_streaming/streaming_client_enhanced/holoscan_client_cloud_streaming/lib/${ARCH_DIR}")

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
    CUDA::cudart
    CUDA::cuda_driver
    "${CLIENT_OPERATOR_SSL_DIR}/libssl.so.3"
    "${CLIENT_OPERATOR_SSL_DIR}/libcrypto.so.3"
    pthread
    dl
    rt
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

# Copy YAML files if they exist
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

# Copy replayer config file if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_replayer.yaml")
    add_custom_command(TARGET streaming_client_demo_enhanced POST_BUILD
        # Copy to target directory
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_replayer.yaml" "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/streaming_client_demo_replayer.yaml"
        # Copy to build root
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_replayer.yaml" "${CMAKE_BINARY_DIR}/streaming_client_demo_replayer.yaml"
    )
endif()

# Copy executable to bin directory
add_custom_command(TARGET streaming_client_demo_enhanced POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:streaming_client_demo_enhanced>" "${CMAKE_BINARY_DIR}/bin/"
)

# Copy SSL libraries from operator to application directory
add_custom_command(TARGET streaming_client_demo_enhanced POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib"
    # Copy SSL libraries
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CLIENT_OPERATOR_SSL_DIR}/libssl.so.3"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib/libssl.so.3"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CLIENT_OPERATOR_SSL_DIR}/libcrypto.so.3"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib/libcrypto.so.3"
    # Copy Holoscan Streaming Stack libraries
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CLIENT_OPERATOR_SSL_DIR}/libStreamingClient.so"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib/libStreamingClient.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CLIENT_OPERATOR_SSL_DIR}/libStreamClientShared.so"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib/libStreamClientShared.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CLIENT_OPERATOR_SSL_DIR}/libNvStreamBase.so"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib/libNvStreamBase.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CLIENT_OPERATOR_SSL_DIR}/libNvStreamingSession.so"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib/libNvStreamingSession.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CLIENT_OPERATOR_SSL_DIR}/libPoco.so"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib/libPoco.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CLIENT_OPERATOR_SSL_DIR}/libcudart.so.12"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib/libcudart.so.12"
    COMMENT "Bundling SSL and streaming libraries for streaming_client_demo_enhanced"
)

# Copy wrapper script and make executable
add_custom_command(TARGET streaming_client_demo_enhanced POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_enhanced_wrapper.sh.in"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/streaming_client_demo_enhanced_wrapper.sh"
    COMMAND chmod +x "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/streaming_client_demo_enhanced_wrapper.sh"
    COMMENT "Installing wrapper script with LD_LIBRARY_PATH for client"
)

# Configure linking to use the bundled SSL libraries
target_link_directories(streaming_client_demo_enhanced PRIVATE "${CLIENT_OPERATOR_SSL_DIR}")

# Set RPATH to prioritize bundled libraries over system libraries
set_target_properties(streaming_client_demo_enhanced PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib:$<TARGET_FILE_DIR:streaming_client_demo_enhanced>:${CLIENT_OPERATOR_SSL_DIR}"
    INSTALL_RPATH "$ORIGIN/lib:$ORIGIN"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    SKIP_BUILD_RPATH FALSE
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
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_replayer.yaml"
    DESTINATION bin
)

# Copy config dependencies for testing
add_custom_target(streaming_client_demo_enhanced_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_client_demo.yaml"
  BYPRODUCTS "streaming_client_demo.yaml"
)
add_dependencies(streaming_client_demo_enhanced streaming_client_demo_enhanced_deps)


