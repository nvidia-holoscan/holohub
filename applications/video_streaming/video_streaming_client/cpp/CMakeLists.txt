# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(streaming_client_demo_enhanced CXX)

# This application is built by the parent add_holohub_application macro

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Detect system architecture
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
  set(ARCH_DIR "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(ARCH_DIR "aarch64")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Find the streaming client operator directory
set(OPERATOR_CLIENT_DIR "${CMAKE_SOURCE_DIR}/operators/video_streaming/streaming_client_enhanced/holoscan_client_cloud_streaming/lib/${ARCH_DIR}")

# Check if NGC binaries exist
if(NOT EXISTS "${OPERATOR_CLIENT_DIR}/libStreamingClient.so")
    message(WARNING "StreamingClient NGC binaries not found at ${OPERATOR_CLIENT_DIR}. Please download them using NGC CLI.")
endif()

add_executable(streaming_client_demo_enhanced
  streaming_client_demo.cpp
)

target_link_libraries(streaming_client_demo_enhanced
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  holoscan::ops::video_stream_replayer
  holoscan::ops::v4l2
  holoscan::ops::format_converter
  holoscan::ops::streaming_client_enhanced
  holoscan::data_loggers::basic_console_logger
  CUDA::cudart
  CUDA::cuda_driver
)

# Copy wrapper script and make executable
add_custom_command(TARGET streaming_client_demo_enhanced POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_enhanced_wrapper.sh.in"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/streaming_client_demo_enhanced_wrapper.sh"
    COMMAND chmod +x "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/streaming_client_demo_enhanced_wrapper.sh"
    COMMENT "Installing wrapper script with LD_LIBRARY_PATH"
)

# Copy config files to multiple locations where they might be needed
add_custom_command(TARGET streaming_client_demo_enhanced POST_BUILD
    # Copy to the binary directory
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml"
        "${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_replayer.yaml"
        "${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_replayer.yaml"
    # Copy to the build root directory (where the application runs)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml"
        "${CMAKE_BINARY_DIR}/streaming_client_demo.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_replayer.yaml"
        "${CMAKE_BINARY_DIR}/streaming_client_demo_replayer.yaml"
    # Copy to the target file directory
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/streaming_client_demo.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_replayer.yaml"
        "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/streaming_client_demo_replayer.yaml"
    COMMENT "Copying config files to all necessary locations"
)

# Set RPATH to prioritize bundled libraries over system libraries
set_target_properties(streaming_client_demo_enhanced PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:streaming_client_demo_enhanced>/lib:$<TARGET_FILE_DIR:streaming_client_demo_enhanced>:${OPERATOR_CLIENT_DIR}"
    INSTALL_RPATH "$ORIGIN/lib:$ORIGIN"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    SKIP_BUILD_RPATH FALSE
)

# Install application and dependencies into the install/ directory for packaging
install(
  TARGETS streaming_client_demo_enhanced
  DESTINATION bin/video_streaming/cpp
)

install(
  FILES streaming_client_demo.yaml streaming_client_demo_replayer.yaml
  DESTINATION bin/video_streaming/cpp
)

install(
  FILES "${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_enhanced_wrapper.sh"
  DESTINATION bin/video_streaming/cpp
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)