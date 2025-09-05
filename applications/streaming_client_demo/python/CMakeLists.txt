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

# Copy streaming_client_demo application file
add_custom_target(python_streaming_client_demo ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_client_demo.py"
  BYPRODUCTS "streaming_client_demo.py"
)

# Copy config files
add_custom_target(python_streaming_client_demo_deps
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_client_demo.yaml"
  BYPRODUCTS "streaming_client_demo.yaml"
)

add_dependencies(python_streaming_client_demo python_streaming_client_demo_deps)

# Add testing
if(BUILD_TESTING)
  # To get the environment path
  find_package(holoscan 1.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME python_streaming_client_demo_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/streaming_client_demo/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "record_type: \"none\"" "record_type: \"visualizer\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "directory: \"./recording_output\"" "directory: \"${RECORDING_DIR}\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "basename: \"streaming_client_demo_output\"" "basename: \"${SOURCE_VIDEO_BASENAME}\"" CONFIG_FILE ${CONFIG_FILE})
  
  # Set test mode parameters
  string(REPLACE "repeat: true" "repeat: false" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "realtime: true" "realtime: false" CONFIG_FILE ${CONFIG_FILE})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_testing.yaml" ${CONFIG_FILE})

  # Add infrastructure test - using streaming client (connection failures are expected in test environment)
  add_test(NAME streaming_client_demo_python_test
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.py
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # Add functional test - using real video data to test video frame processing
  add_test(NAME streaming_client_demo_python_functional_test
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_functional_test.sh 
                   ${CMAKE_CURRENT_BINARY_DIR} 
                   ${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo_functional.py
                   ${HOLOHUB_DATA_DIR}/endoscopy
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # Set environment for both tests
  set_property(TEST streaming_client_demo_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib:${CMAKE_CURRENT_BINARY_DIR}/python/lib")
  set_property(TEST streaming_client_demo_python_functional_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib:${CMAKE_CURRENT_BINARY_DIR}/python/lib")

  # Infrastructure test properties
  set_tests_properties(streaming_client_demo_python_test PROPERTIES
                DEPENDS python_streaming_client_demo_deps
                TIMEOUT 90
                PASS_REGULAR_EXPRESSION "Test PASSED: StreamingClient functionality validated successfully"
                FAIL_REGULAR_EXPRESSION "Test FAILED:")

  # Functional test properties  
  set_tests_properties(streaming_client_demo_python_functional_test PROPERTIES
                DEPENDS python_streaming_client_demo_deps
                TIMEOUT 120
                PASS_REGULAR_EXPRESSION "FUNCTIONAL test.*successful"
                FAIL_REGULAR_EXPRESSION "FUNCTIONAL test.*FAILED")

  # Frame validation test disabled - no recording in current streaming client setup
  # TODO: Re-enable when recording functionality is implemented
endif()

# Install application and dependencies into the install/ directory for packaging
install(
  FILES streaming_client_demo.py
  DESTINATION bin/streaming_client_demo/python
)

install(
  FILES streaming_client_demo.yaml
  DESTINATION bin/streaming_client_demo/python
)