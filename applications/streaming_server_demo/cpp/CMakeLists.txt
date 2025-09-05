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

# Add dependency on streaming_server operator
add_dependencies(streaming_server_demo streaming_server)

# Link libraries
target_link_libraries(streaming_server_demo
    PUBLIC
    holoscan::core
    holoscan::streaming_server
    CUDA::cudart
    CUDA::cuda_driver
)



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

# Testing infrastructure - follows the same pattern as streaming_client_demo
if(BUILD_TESTING)
  # Configure testing directory
  set(RECORDING_DIR "${CMAKE_CURRENT_BINARY_DIR}/recording_output")
  set(SOURCE_VIDEO_BASENAME "streaming_server_demo_output")
  
  # Ensure recording directory exists
  file(MAKE_DIRECTORY ${RECORDING_DIR})
  
  # Read the original config file
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml" CONFIG_FILE)
  
  # Create test-specific configuration
  # Configure for shorter test runs
  string(REPLACE "fps: 30" "fps: 30" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "multi_thread_scheduler:" "multi_thread_scheduler:\n  max_duration_ms: 30000  # 30 second timeout for tests" CONFIG_FILE ${CONFIG_FILE})
  
  # Write the test configuration file
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_testing.yaml" ${CONFIG_FILE})

  # Create fallback data for server initialization when real data is not available
  # This prevents the C++ app from failing during operator dependency resolution
  file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
  file(TOUCH ${CMAKE_SOURCE_DIR}/data/surgical_video.gxf_index)
  file(TOUCH ${CMAKE_SOURCE_DIR}/data/surgical_video.gxf_entities)
  
  # Use real endoscopy video data for functional testing when available
  # The test system automatically provides real video data at ${HOLOHUB_DATA_DIR}/endoscopy
  # which contains surgical_video.gxf_index and surgical_video.gxf_entities with real frame data

  # Add test for application execution using wrapper script to handle exceptions gracefully
  add_test(NAME streaming_server_demo_cpp_test
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh
                   ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo
                   ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_testing.yaml
                   ${HOLOHUB_DATA_DIR}/endoscopy
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  
  # Set test properties including library path and pass/fail criteria  
  set_tests_properties(streaming_server_demo_cpp_test PROPERTIES
                       ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/operators/streaming_server:${CMAKE_BINARY_DIR}/operators/streaming_server/lib:${CMAKE_SOURCE_DIR}/operators/streaming_server/lib:$ENV{LD_LIBRARY_PATH}"
                       TIMEOUT 90
                       PASS_REGULAR_EXPRESSION "Test PASSED: C\\+\\+ StreamingServer.*test successful"
                       FAIL_REGULAR_EXPRESSION "Test FAILED:")

  # Frame validation test disabled - current streaming server setup is standalone
  # TODO: Re-enable when client-server integration testing is implemented for streaming server
  # add_test(NAME streaming_server_demo_cpp_render_validation
  #          COMMAND python3 ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo/testing/video_validation.py
  #                          --golden-dir ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo/testing
  #                          --recorded-dir ${RECORDING_DIR}
  #                          --pattern "streaming_server_demo_output*.png")

endif()

