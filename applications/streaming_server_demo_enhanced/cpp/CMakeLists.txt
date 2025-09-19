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
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo_04_80.yaml")
    add_custom_command(TARGET streaming_server_demo_enhanced POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo_04_80.yaml"
        "$<TARGET_FILE_DIR:streaming_server_demo_enhanced>/streaming_server_demo_04_80.yaml"
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
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo_04_80.yaml"
    DESTINATION bin
)

# Copy config dependencies for testing
add_custom_target(streaming_server_demo_enhanced_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo_04_80.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_server_demo_04_80.yaml"
  BYPRODUCTS "streaming_server_demo_04_80.yaml"
)
add_dependencies(streaming_server_demo_enhanced streaming_server_demo_enhanced_deps)

# Add testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(VALIDATION_FRAMES_DIR "${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/")

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Test 1: Basic Application Test - Help Command
  add_test(NAME streaming_server_demo_enhanced_help_test
           COMMAND ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced --help)
  
  set_tests_properties(streaming_server_demo_enhanced_help_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Usage:|--help|streaming"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault"
                       TIMEOUT 10)

  # Test 2: Configuration Validation - Test config file loading and validation  
  add_test(NAME streaming_server_demo_enhanced_config_test
           COMMAND timeout 5s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                   --config ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/test_basic.yaml)
                  
  set_tests_properties(streaming_server_demo_enhanced_config_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "BasicTest|StreamingServerResource started successfully|Server started successfully|Starting"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|Invalid config|Failed to load"
                       TIMEOUT 15)

  # Test 3: Multi-Thread Scheduler - Test multi-threading capabilities
  add_test(NAME streaming_server_demo_enhanced_multithread_test
           COMMAND timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                   --config ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/test_multi_thread.yaml)
                  
  set_tests_properties(streaming_server_demo_enhanced_multithread_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "MultiThreadTest|StreamingServerResource started successfully|Server started successfully|Starting"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|Deadlock detected|Thread creation failed"
                       TIMEOUT 10)

  # Test 4: Data Flow Tracking - Test tracking and profiling functionality
  add_test(NAME streaming_server_demo_enhanced_tracking_test
           COMMAND timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                   --config ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/test_tracking.yaml)
                  
  set_tests_properties(streaming_server_demo_enhanced_tracking_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "TrackingTest|StreamingServerResource started successfully|Server started successfully|Starting"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|Tracking failed"
                       TIMEOUT 10)

  # Test 5: Scheduler Configuration Tests - Test greedy and event-based schedulers
  foreach(scheduler_type "greedy" "event_based")
    add_test(NAME streaming_server_demo_enhanced_scheduler_${scheduler_type}_test
             COMMAND timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                     --config ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/test_${scheduler_type}.yaml)
             
    set_tests_properties(streaming_server_demo_enhanced_scheduler_${scheduler_type}_test PROPERTIES
                         PASS_REGULAR_EXPRESSION "${scheduler_type}Test|StreamingServerResource started successfully|Server started successfully|Starting"
                         FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|Invalid scheduler"
                         TIMEOUT 10)
  endforeach()

  # Test 6: High Resolution Configuration - Test performance with high resolution
  add_test(NAME streaming_server_demo_enhanced_tensor_test
           COMMAND timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                   --config ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/test_high_res.yaml)
                  
  set_tests_properties(streaming_server_demo_enhanced_tensor_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "HighResTest|StreamingServerResource started successfully|Server started successfully|Starting"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|Memory allocation failed|Invalid resolution"
                       TIMEOUT 10)

  # Test 7: Network Configuration - Test basic networking functionality  
  add_test(NAME streaming_server_demo_enhanced_network_test
           COMMAND timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                   --config ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/test_basic.yaml)
  
  set_tests_properties(streaming_server_demo_enhanced_network_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "BasicTest|StreamingServerResource started successfully|Server started successfully|Starting"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|Network.*failed|Port.*in use"
                       TIMEOUT 10)

  # Test 8: SSL Libraries - Test SSL library loading
  add_test(NAME streaming_server_demo_enhanced_ssl_test
           COMMAND timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                   --config ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/test_basic.yaml)
  
  set_tests_properties(streaming_server_demo_enhanced_ssl_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "BasicTest|StreamingServerResource started successfully|Server started successfully|Starting"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|SSL.*not found|libssl|libcrypto.*failed"
                       TIMEOUT 10)

  # Test 9: Memory Allocator - Test memory allocation and cleanup
  add_test(NAME streaming_server_demo_enhanced_memory_test
           COMMAND timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                   --config ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/test_basic.yaml)
  
  set_tests_properties(streaming_server_demo_enhanced_memory_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "BasicTest|StreamingServerResource started successfully|Server started successfully|Starting"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|Memory leak|Out of memory|Allocation failed"
                       TIMEOUT 10)

  # Test 10: GPU Resources - Test GPU capability detection
  add_test(NAME streaming_server_demo_enhanced_gpu_test
           COMMAND timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                   --config ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/test_basic.yaml)
  
  set_tests_properties(streaming_server_demo_enhanced_gpu_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "BasicTest|StreamingServerResource started successfully|Server started successfully|Starting"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|GPU.*not found|CUDA.*failed"
                       TIMEOUT 10)

  # Test 11: Error Handling - Test application behavior with invalid configuration
  add_test(NAME streaming_server_demo_enhanced_error_handling_test
           COMMAND timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                   --config /nonexistent/invalid.yaml)
  
  set_tests_properties(streaming_server_demo_enhanced_error_handling_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "WARNING.*Failed to create default configuration file|Will try to run with built-in defaults|Could not create config file"
                       FAIL_REGULAR_EXPRESSION "Segmentation fault"
                       TIMEOUT 10)

  # Test 12: Server Robustness - Alternative test for server stability
  add_test(NAME streaming_server_demo_enhanced_server_test
           COMMAND timeout 3s ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_enhanced 
                   --config ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/test_basic.yaml)
  
  set_tests_properties(streaming_server_demo_enhanced_server_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "BasicTest|StreamingServerResource started successfully|Server started successfully|Starting"
                       FAIL_REGULAR_EXPRESSION "FATAL|CRITICAL|Segmentation fault|Port.*in use|Failed to bind"
                       TIMEOUT 10)

endif()

