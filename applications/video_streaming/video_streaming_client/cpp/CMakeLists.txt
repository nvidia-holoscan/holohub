# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(video_streaming_client_demo_cpp CXX)

find_package(holoscan 3.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Find required packages
find_package(CUDAToolkit REQUIRED)

# Detect system architecture
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
  set(ARCH_DIR "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "arm64")
  set(ARCH_DIR "aarch64")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

# Find the streaming client operator directory using relative path
set(OPERATOR_CLIENT_DIR "${CMAKE_SOURCE_DIR}/operators/video_streaming/video_streaming_client/holoscan_client_cloud_streaming/lib/${ARCH_DIR}")

# Validate operator bundle exists - fail early with clear error message
if(NOT EXISTS "${OPERATOR_CLIENT_DIR}")
  message(FATAL_ERROR "Operator client directory not found: ${OPERATOR_CLIENT_DIR}\n"
                      "Please ensure NGC assets are installed per README:\n"
                      "  cd operators/video_streaming/video_streaming_client\n"
                      "  ngc registry resource download-version \"nvidia/holoscan_client_cloud_streaming:0.2\"\n"
                      "  unzip holoscan_client_cloud_streaming_v0.2/holoscan_client_cloud_streaming.zip -d holoscan_client_cloud_streaming")
endif()

add_executable(video_streaming_client_demo_cpp
  streaming_client_demo.cpp
)

target_link_libraries(video_streaming_client_demo_cpp
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  holoscan::ops::video_stream_replayer
  holoscan::ops::v4l2
  holoscan::ops::format_converter
  holoscan::ops::video_streaming_client
  holoscan::data_loggers::basic_console_logger
  CUDA::cudart
  CUDA::cuda_driver
)

# Copy wrapper script and make executable
add_custom_command(TARGET video_streaming_client_demo_cpp POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_wrapper.sh"
        "$<TARGET_FILE_DIR:video_streaming_client_demo_cpp>/streaming_client_demo_wrapper.sh"
    COMMAND chmod +x "$<TARGET_FILE_DIR:video_streaming_client_demo_cpp>/streaming_client_demo_wrapper.sh"
    COMMENT "Installing wrapper script with LD_LIBRARY_PATH"
)

# Copy config files to necessary locations
add_custom_command(TARGET video_streaming_client_demo_cpp POST_BUILD
    # Copy to the binary directory (for local testing)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml"
        "${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_replayer.yaml"
        "${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_replayer.yaml"
    # Copy to the build root directory (where holohub run executes)
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml"
        "${CMAKE_BINARY_DIR}/streaming_client_demo.yaml"
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_replayer.yaml"
        "${CMAKE_BINARY_DIR}/streaming_client_demo_replayer.yaml"
    COMMENT "Copying config files"
)

# Set RPATH to prioritize bundled libraries over system libraries
# BUILD_RPATH: Uses absolute path to NGC binaries in source tree (for development)
# INSTALL_RPATH: Uses relative paths to find NGC binaries installed at <prefix>/lib
set_target_properties(video_streaming_client_demo_cpp PROPERTIES
    BUILD_RPATH "$<TARGET_FILE_DIR:video_streaming_client_demo_cpp>/lib:$<TARGET_FILE_DIR:video_streaming_client_demo_cpp>:${OPERATOR_CLIENT_DIR}"
    INSTALL_RPATH "$ORIGIN/lib:$ORIGIN:$ORIGIN/../../../lib"
    BUILD_WITH_INSTALL_RPATH FALSE
    INSTALL_RPATH_USE_LINK_PATH FALSE
    SKIP_BUILD_RPATH FALSE
)

# Install application and dependencies into the install/ directory for packaging
install(
  TARGETS video_streaming_client_demo_cpp
  DESTINATION bin/video_streaming_client/cpp
)

install(
  FILES streaming_client_demo.yaml streaming_client_demo_replayer.yaml
  DESTINATION bin/video_streaming_client/cpp
)

install(
  FILES "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_wrapper.sh"
  DESTINATION bin/video_streaming_client/cpp
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)