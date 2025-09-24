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
project(streaming_server_demo_enhanced CXX)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# Bundle NEWER SSL libraries from streaming_server_enhanced operator (supports OpenSSL 3.0-3.5)
# Detect system architecture
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
  set(ARCH_DIR "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(ARCH_DIR "aarch64")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
set(OPERATOR_SSL_DIR "${CMAKE_SOURCE_DIR}/operators/streaming_server_enhanced/holoscan_server_cloud_streaming/lib/${ARCH_DIR}")

# Add executable
add_executable(streaming_server_demo_enhanced
    streaming_server_demo.cpp
)

# Link libraries - include explicit SSL library paths from operator
target_link_libraries(streaming_server_demo_enhanced
    PRIVATE
    holoscan::core
    holoscan::ops::streaming_server_enhanced
    ${OpenCV_LIBS}
    CUDA::cudart
    "${OPERATOR_SSL_DIR}/libssl.so.3"
    "${OPERATOR_SSL_DIR}/libcrypto.so.3"
    pthread
    dl
    rt
)

# Set compile definitions
target_compile_definitions(streaming_server_demo_enhanced
    PRIVATE
    HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
    HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
    HOLOSCAN_PATCH_VERSION=${holoscan_VERSION_PATCH}
)

# Copy YAML file if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml")
    add_custom_command(TARGET streaming_server_demo_enhanced POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/streaming_server_demo.yaml"
    )
endif()

# Copy ALL required libraries from operator to application directory
add_custom_command(TARGET streaming_server_demo_enhanced POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/lib"
    # Copy SSL libraries (newer versions)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libssl.so.3"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/lib/libssl.so.3"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libcrypto.so.3"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/lib/libcrypto.so.3"
    # Copy Holoscan Streaming Stack libraries (required at runtime)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libStreamingServer.so"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/lib/libStreamingServer.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libNvStreamServer.so"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/lib/libNvStreamServer.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libNvStreamBase.so"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/lib/libNvStreamBase.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libNvStreamingSession.so"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/lib/libNvStreamingSession.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libPoco.so"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/lib/libPoco.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libcudart.so.12"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/lib/libcudart.so.12"
    COMMENT "Bundling ALL required libraries from streaming_server_enhanced operator"
)

# Configure linking to use the bundled newer SSL libraries
target_link_directories(streaming_server_demo_enhanced PRIVATE "${OPERATOR_SSL_DIR}")

# Copy wrapper script and make executable
add_custom_command(TARGET streaming_server_demo_enhanced POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo_enhanced_wrapper.sh.in"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/streaming_server_demo_enhanced_wrapper.sh"
    COMMAND chmod +x "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/streaming_server_demo_enhanced_wrapper.sh"
    COMMENT "Installing wrapper script with LD_LIBRARY_PATH"
)

# Set RPATH to prioritize bundled libraries over system libraries
# Use BUILD_RPATH so it works during development/testing  
set_target_properties(streaming_server_demo_enhanced PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/lib:$<TARGET_FILE_DIR:streaming_server_demo_enhanced>:${OPERATOR_SSL_DIR}"
    INSTALL_RPATH "$ORIGIN/lib:$ORIGIN"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    SKIP_BUILD_RPATH FALSE
)

# Install rules
install(
    TARGETS streaming_server_demo_enhanced              
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Install config files
install(
    FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml"
    DESTINATION bin
)

# Copy config dependencies for testing
add_custom_target(streaming_server_demo_enhanced_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_server_demo.yaml"
  BYPRODUCTS "streaming_server_demo.yaml"
)
add_dependencies(streaming_server_demo_enhanced streaming_server_demo_enhanced_deps)


