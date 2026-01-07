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

cmake_minimum_required(VERSION 3.20)
project(video_streaming_server_demo CXX)

# This application is built by the parent add_holohub_application macro

find_package(holoscan 3.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Bundle NEWER SSL libraries from video_streaming_server operator (supports OpenSSL 3.0-3.5)
# Detect system architecture
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
  set(ARCH_DIR "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(ARCH_DIR "aarch64")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()
# Find the streaming server operator directory using relative path
set(OPERATOR_SSL_DIR "${CMAKE_SOURCE_DIR}/operators/video_streaming/video_streaming_server/holoscan_server_cloud_streaming/lib/${ARCH_DIR}")

# Validate operator bundle exists - fail early with clear error message
if(NOT EXISTS "${OPERATOR_SSL_DIR}")
  message(FATAL_ERROR "Operator SSL directory not found: ${OPERATOR_SSL_DIR}\n"
                      "Please ensure NGC assets are installed per README:\n"
                      "  cd operators/video_streaming/video_streaming_server\n"
                      "  ngc registry resource download-version \"nvidia/holoscan_server_cloud_streaming:0.2\"\n"
                      "  unzip holoscan_server_cloud_streaming_v0.2/holoscan_server_cloud_streaming.zip -d holoscan_server_cloud_streaming")
endif()

# Add executable
add_executable(video_streaming_server_demo
    video_streaming_server_demo.cpp
)

# Link libraries
target_link_libraries(video_streaming_server_demo
    PRIVATE
    holoscan::core
    holoscan::ops::video_streaming_server
    CUDA::cudart
    pthread
    dl
    rt
)

# Set compile definitions
target_compile_definitions(video_streaming_server_demo
    PRIVATE
    HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
    HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
    HOLOSCAN_PATCH_VERSION=${holoscan_VERSION_PATCH}
)

# Copy YAML file if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_server_demo.yaml")
    add_custom_command(TARGET video_streaming_server_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_server_demo.yaml"
        "$<TARGET_FILE_DIR:video_streaming_server_demo>/video_streaming_server_demo.yaml"
    )
endif()

# Copy NGC streaming libraries from operator to application directory
# Note: OpenSSL and CUDA are provided by system (via RPATH), not bundled
add_custom_command(TARGET video_streaming_server_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:video_streaming_server_demo>/lib"
    # Copy Holoscan Streaming Stack libraries (required at runtime)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libStreamingServer.so"
        "$<TARGET_FILE_DIR:video_streaming_server_demo>/lib/libStreamingServer.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libNvStreamServer.so"
        "$<TARGET_FILE_DIR:video_streaming_server_demo>/lib/libNvStreamServer.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libNvStreamBase.so"
        "$<TARGET_FILE_DIR:video_streaming_server_demo>/lib/libNvStreamBase.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libNvStreamingSession.so"
        "$<TARGET_FILE_DIR:video_streaming_server_demo>/lib/libNvStreamingSession.so"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${OPERATOR_SSL_DIR}/libPoco.so"
        "$<TARGET_FILE_DIR:video_streaming_server_demo>/lib/libPoco.so"
    COMMENT "Bundling NGC streaming libraries from video_streaming_server operator"
)

# Copy wrapper script and make executable
add_custom_command(TARGET video_streaming_server_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_server_demo_wrapper.sh"
        "$<TARGET_FILE_DIR:video_streaming_server_demo>/video_streaming_server_demo_wrapper.sh"
    COMMAND chmod +x "$<TARGET_FILE_DIR:video_streaming_server_demo>/video_streaming_server_demo_wrapper.sh"
    COMMENT "Installing wrapper script with LD_LIBRARY_PATH"
)

# Set RPATH to prioritize system libraries for OpenSSL and CUDA
# Priority: System OpenSSL 3.4 > Holoscan libs (UCX) > system CUDA > bundled NGC libs
set_target_properties(video_streaming_server_demo PROPERTIES
    BUILD_RPATH "/usr/local/ssl/lib64:/opt/nvidia/holoscan/lib:/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:$<TARGET_FILE_DIR:video_streaming_server_demo>/lib:$<TARGET_FILE_DIR:video_streaming_server_demo>:${OPERATOR_SSL_DIR}"
    INSTALL_RPATH "/usr/local/ssl/lib64:/opt/nvidia/holoscan/lib:/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:$ORIGIN/lib:$ORIGIN:$ORIGIN/../../../lib"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    SKIP_BUILD_RPATH FALSE
)

# Copy config dependencies for testing
add_custom_target(video_streaming_server_demo_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_server_demo.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "video_streaming_server_demo.yaml"
  BYPRODUCTS "video_streaming_server_demo.yaml"
)
add_dependencies(video_streaming_server_demo video_streaming_server_demo_deps)

# Install application and dependencies into the install/ directory for packaging
install(
  TARGETS video_streaming_server_demo
  DESTINATION bin/video_streaming_server/cpp
)

install(
  FILES video_streaming_server_demo.yaml
  DESTINATION bin/video_streaming_server/cpp
)

# Install wrapper script
install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_server_demo_wrapper.sh"
  DESTINATION bin/video_streaming_server/cpp
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install bundled libraries (OpenSSL 3.x and NGC streaming libraries)
install(
  DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib/"
  DESTINATION bin/video_streaming_server/cpp/lib
  FILES_MATCHING PATTERN "*.so*"
)
