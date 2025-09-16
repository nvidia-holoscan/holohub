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
message(STATUS "=== PROCESSING streaming_client_demo_04_80_streaming/cpp/CMakeLists.txt ===")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(holoscan REQUIRED)

# Add executable
add_executable(streaming_client_demo_04_80_streaming
    streaming_client_demo.cpp
)

# Print success message
message(STATUS "✓ Executable target 'streaming_client_demo_04_80_streaming' created")

# Set build directory
message(STATUS "Build streaming_client_demo_04_80_streaming directory: ${CMAKE_BINARY_DIR}/streaming_client_demo_04_80_streaming")

# Set compile definitions
target_compile_definitions(streaming_client_demo_04_80_streaming
    PRIVATE
    HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
    HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
    HOLOSCAN_PATCH_VERSION=${holoscan_VERSION_PATCH}
)

# Set compile features
target_compile_features(streaming_client_demo_04_80_streaming
    PUBLIC
    cxx_std_17
)

# Set compile options
target_compile_options(streaming_client_demo_04_80_streaming
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-reorder>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
)

# Add include directories
target_include_directories(streaming_client_demo_04_80_streaming 
    PUBLIC 
    $<BUILD_INTERFACE:${OpenCV_INCLUDE_DIRS}>
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link libraries
target_link_libraries(streaming_client_demo_04_80_streaming
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
message(STATUS "✓ Libraries linked to streaming_client_demo_04_80_streaming")

# Get target properties
get_target_property(TARGET_TYPE streaming_client_demo_04_80_streaming TYPE)
message(STATUS "Target type: ${TARGET_TYPE}")

get_target_property(TARGET_SOURCES streaming_client_demo_04_80_streaming SOURCES)
message(STATUS "Target sources: ${TARGET_SOURCES}")

message(STATUS "=== COMPLETED streaming_client_demo_04_80_streaming/cpp/CMakeLists.txt ===")

# Copy libraries
add_custom_command(TARGET streaming_client_demo_04_80_streaming PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:streaming_client_demo_04_80_streaming>/lib"
)

# Copy executable and config files
add_custom_command(TARGET streaming_client_demo_04_80_streaming POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin"
)

# Copy YAML file if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml")
    add_custom_command(TARGET streaming_client_demo_04_80_streaming POST_BUILD
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
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "$<TARGET_FILE_DIR:streaming_client_demo_04_80_streaming>/streaming_client_demo.yaml"
    )
endif()

# Copy executable to bin directory
add_custom_command(TARGET streaming_client_demo_04_80_streaming POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:streaming_client_demo_04_80_streaming>" "${CMAKE_BINARY_DIR}/bin/"
)

# Install rules
install(
    TARGETS streaming_client_demo_04_80_streaming
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
add_custom_target(streaming_client_demo_04_80_streaming_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_client_demo.yaml"
  BYPRODUCTS "streaming_client_demo.yaml"
)
add_dependencies(streaming_client_demo_04_80_streaming streaming_client_demo_04_80_streaming_deps)

# Add testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(VALIDATION_FRAMES_DIR "${CMAKE_SOURCE_DIR}/applications/streaming_client_demo_04_80_streaming/testing/")
  set(TEST_CONFIG_DIR "${CMAKE_SOURCE_DIR}/applications/streaming_client_demo_04_80_streaming/testing/")

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Test 1: Help Command Test (keep one help test for CLI validation)
  add_test(NAME streaming_client_demo_04_80_streaming_help_test
           COMMAND streaming_client_demo_04_80_streaming --help)

  set_tests_properties(streaming_client_demo_04_80_streaming_help_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Usage:|--help|streaming"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault"
                       TIMEOUT 10)

  # Test 2: Basic Configuration Test - Functional test with minimal config
  add_test(NAME streaming_client_demo_04_80_streaming_config_test
           COMMAND timeout 15s ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_04_80_streaming 
                   --config "${TEST_CONFIG_DIR}/config_basic_test.yaml"
                   --data "/workspace/holohub/data/endoscopy")
                   
  set_tests_properties(streaming_client_demo_04_80_streaming_config_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Streaming Client Test Application|Successfully loaded configuration|Configuration:"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|does not exist"
                       TIMEOUT 20)

  # Test 3: Data Directory Discovery Test - Test data path validation
  add_test(NAME streaming_client_demo_04_80_streaming_data_test
           COMMAND timeout 10s ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_04_80_streaming 
                   --config "${TEST_CONFIG_DIR}/config_basic_test.yaml"
                   --data "/workspace/holohub/data/endoscopy")
                   
  set_tests_properties(streaming_client_demo_04_80_streaming_data_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Using data from:|Video file path:|Current working directory"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|does not exist"
                       TIMEOUT 15)

  # Test 4: Configuration Validation Test - Invalid config handling
  add_test(NAME streaming_client_demo_04_80_streaming_memory_test
           COMMAND timeout 5s ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_04_80_streaming 
                   --config "${TEST_CONFIG_DIR}/config_error_test.yaml"
                   --data "/workspace/holohub/data/endoscopy")
                   
  set_tests_properties(streaming_client_demo_04_80_streaming_memory_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Streaming Client Test Application|Successfully loaded configuration|Configuration:"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault"
                       TIMEOUT 10)

  # Test 5: Scheduler Configuration Tests - Functional tests with different schedulers
  foreach(scheduler_type "default" "greedy" "multi_thread" "event_based")
    # Create a scheduler-specific config file
    configure_file("${TEST_CONFIG_DIR}/config_scheduler_test.yaml" 
                   "${CMAKE_CURRENT_BINARY_DIR}/config_scheduler_${scheduler_type}_test.yaml" 
                   COPYONLY)
    
    add_test(NAME streaming_client_demo_04_80_streaming_scheduler_${scheduler_type}_test
             COMMAND timeout 15s ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_04_80_streaming 
                     --config "${CMAKE_CURRENT_BINARY_DIR}/config_scheduler_${scheduler_type}_test.yaml"
                     --data "/workspace/holohub/data/endoscopy")
             
    set_tests_properties(streaming_client_demo_04_80_streaming_scheduler_${scheduler_type}_test PROPERTIES
                         PASS_REGULAR_EXPRESSION "Streaming Client Test Application|Configuration:|Successfully loaded"
                         FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault"
                         TIMEOUT 20)
  endforeach()

  # Test 6: V4L2 Camera Source Test - Functional test (graceful handling of missing camera)
  add_test(NAME streaming_client_demo_04_80_streaming_v4l2_test
           COMMAND timeout 10s ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_04_80_streaming 
                   --config "${TEST_CONFIG_DIR}/config_v4l2_test.yaml"
                   --data "/workspace/holohub/data/endoscopy")
                   
  set_tests_properties(streaming_client_demo_04_80_streaming_v4l2_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Streaming Client Test Application|Configuration:|Successfully loaded|Warning"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault"
                       TIMEOUT 15)

  # Test 7: Format Converter Test - Test with basic config
  add_test(NAME streaming_client_demo_04_80_streaming_format_test
           COMMAND timeout 15s ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_04_80_streaming 
                   --config "${TEST_CONFIG_DIR}/config_basic_test.yaml"
                   --data "/workspace/holohub/data/endoscopy")
                   
  set_tests_properties(streaming_client_demo_04_80_streaming_format_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Streaming Client Test Application|Format converter|Successfully loaded"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault"
                       TIMEOUT 20)

  # Test 8: Network Configuration Test - Test with disabled streaming
  add_test(NAME streaming_client_demo_04_80_streaming_network_test
           COMMAND timeout 10s ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_04_80_streaming 
                   --config "${TEST_CONFIG_DIR}/config_basic_test.yaml"
                   --data "/workspace/holohub/data/endoscopy")
  
  set_tests_properties(streaming_client_demo_04_80_streaming_network_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Server: 127.0.0.1:48010|Receive frames: no|Send frames: no"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault"
                       TIMEOUT 15)

  # Test 9: Application Startup Test - Test application initialization
  add_test(NAME streaming_client_demo_04_80_streaming_gpu_test
           COMMAND timeout 10s ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_04_80_streaming 
                   --config "${TEST_CONFIG_DIR}/config_basic_test.yaml"
                   --data "/workspace/holohub/data/endoscopy")
  
  set_tests_properties(streaming_client_demo_04_80_streaming_gpu_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Streaming Client Test Application|Resolution: 640x480|FPS: 30"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault"
                       TIMEOUT 15)

  # Test 10: Error Handling Test - Test with invalid configuration path
  # This test expects the app to show warnings about missing config and then gracefully fail
  add_test(NAME streaming_client_demo_04_80_streaming_logging_test
           COMMAND bash -c "timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_04_80_streaming --config '/nonexistent/invalid.yaml' --data '/workspace/holohub/data/endoscopy' 2>&1 | grep -q 'WARNING.*Failed to create default configuration file' && echo 'TEST_PASSED: Config error handling working'")
  
  set_tests_properties(streaming_client_demo_04_80_streaming_logging_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "TEST_PASSED.*Config error handling working"
                       TIMEOUT 10)

endif()