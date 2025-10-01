# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(streaming_client_demo_enhanced CXX)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(streaming_client_demo_enhanced
  main.cpp
)

target_link_libraries(streaming_client_demo_enhanced
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  holoscan::ops::video_stream_replayer
  holoscan::ops::v4l2
  holoscan::ops::format_converter
  holoscan::ops::video_streaming
  holoscan::data_loggers::basic_console_logger
  CUDA::cudart
  CUDA::cuda_driver
)

# Copy the config to the binary directory
add_custom_target(streaming_client_demo_enhanced_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_replayer.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_client_demo_replayer.yaml"
  BYPRODUCTS "streaming_client_demo_replayer.yaml"
)
add_dependencies(streaming_client_demo_enhanced streaming_client_demo_enhanced_deps)

# Add testing
if(BUILD_TESTING)
  add_test(NAME streaming_client_demo_enhanced_cpp_test
           COMMAND streaming_client_demo_enhanced
                   --config ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_replayer.yaml
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(streaming_client_demo_enhanced_cpp_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "StreamingClient created successfully"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()

# Install application and dependencies into the install/ directory for packaging
install(
  TARGETS streaming_client_demo_enhanced
  DESTINATION bin/video_streaming_demo_enhanced/cpp
)

install(
  FILES streaming_client_demo_replayer.yaml
  DESTINATION bin/video_streaming_demo_enhanced/cpp
)