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

# Copy streaming_server_demo application file to both locations
add_custom_target(python_streaming_server_demo ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.py" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.py" ${CMAKE_BINARY_DIR}
  DEPENDS "streaming_server_demo.py"
  BYPRODUCTS "streaming_server_demo.py"
)

# Install application and dependencies into the install/ directory for packaging
install(
  FILES streaming_server_demo.py
  DESTINATION bin
)

# Testing infrastructure - follows the same pattern as streaming_client_demo
if(BUILD_TESTING)
  # Configure testing directory
  set(RECORDING_DIR "${CMAKE_CURRENT_BINARY_DIR}/recording_output")
  set(SOURCE_VIDEO_BASENAME "streaming_server_demo_output")
  
  # Ensure recording directory exists
  file(MAKE_DIRECTORY ${RECORDING_DIR})
  
  # Create test-specific YAML configuration
  set(CONFIG_FILE "# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

%YAML 1.2
---
# Application configuration for testing
application:
  title: Streaming Server Test App
  version: 1.0
  log_level: INFO

# Streaming server settings - standalone mode for testing
streaming_server:
  # Video/stream parameters
  width: 854
  height: 480
  fps: 30
  
  # Server connection settings
  port: 48010
  multi_instance: false
  server_name: \"StreamingServerTest\"
  
  # Operation mode - Bidirectional streaming (receive from client, send back)
  receive_frames: true
  send_frames: true
  allocator: !ref \"allocator\"

# Memory allocator configuration
allocator:
  type: \"holoscan::UnboundedAllocator\"

# Use greedy scheduler for testing reliability
scheduler: \"greedy\"

# Greedy scheduler configuration for tests
greedy_scheduler:
  max_duration_ms: 30000  # 30 second timeout for tests
  check_recession_period_ms: 100

# Enable data flow tracking for debugging
tracking: false

# Logging configuration
log_level: \"INFO\"
")

  # Write the test configuration
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_testing.yaml" ${CONFIG_FILE})

  # Add infrastructure test for Python streaming server application
  add_test(NAME streaming_server_demo_python_test
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh
                   ${CMAKE_CURRENT_BINARY_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.py
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  
  # Set test properties for Python infrastructure test
  set_tests_properties(streaming_server_demo_python_test PROPERTIES
                       ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
                       TIMEOUT 90
                       PASS_REGULAR_EXPRESSION "Test PASSED: StreamingServer functionality validated successfully"
                       FAIL_REGULAR_EXPRESSION "Test FAILED:")

  # Add functional test for Python streaming server with real video data
  add_test(NAME streaming_server_demo_python_functional_test
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_functional_test.sh
                   ${CMAKE_CURRENT_BINARY_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo_functional.py
                   ${HOLOHUB_DATA_DIR}/endoscopy
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  
  # Set test properties for Python functional test
  set_tests_properties(streaming_server_demo_python_functional_test PROPERTIES
                       ENVIRONMENT "PYTHONPATH=${CMAKE_CURRENT_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
                       TIMEOUT 120
                       PASS_REGULAR_EXPRESSION "FUNCTIONAL test PASSED: StreamingServer.*successful"
                       FAIL_REGULAR_EXPRESSION "FUNCTIONAL test.*FAILED")

  # Frame validation test disabled - current streaming server setup is standalone
  # TODO: Re-enable when client-server integration testing is implemented
  # add_test(NAME streaming_server_demo_python_render_validation
  #          COMMAND python3 ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo/testing/video_validation.py
  #                          --golden-dir ${CMAKE_SOURCE_DIR}/applications/streaming_server_demo/testing
  #                          --recorded-dir ${RECORDING_DIR}
  #                          --pattern "streaming_server_demo_output*.png")

endif()
