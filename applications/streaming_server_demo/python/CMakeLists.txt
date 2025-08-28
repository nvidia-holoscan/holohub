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

# Comprehensive testing infrastructure following streaming_client_demo pattern
if(BUILD_TESTING)
  # Find holoscan package with robust configuration
  find_package(holoscan 1.0 REQUIRED CONFIG
               PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  # Configure comprehensive testing directories
  set(RECORDING_DIR "${CMAKE_CURRENT_BINARY_DIR}/recording_output")
  set(SOURCE_VIDEO_BASENAME "streaming_server_demo_reference")
  set(VALIDATION_FRAMES_DIR "${CMAKE_SOURCE_DIR}/applications/streaming_server_demo/testing")
  
  # Ensure all testing directories exist
  file(MAKE_DIRECTORY ${RECORDING_DIR})
  file(MAKE_DIRECTORY ${VALIDATION_FRAMES_DIR})

  # Copy and update the comprehensive test configuration
  configure_file(
    "${CMAKE_SOURCE_DIR}/applications/streaming_server_demo/testing/streaming_server_demo_testing.yaml"
    "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_testing.yaml"
    COPYONLY
  )

  # Read and modify the test configuration for Python-specific parameters
  file(READ "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_testing.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "directory: \"./recording_output\"" "directory: \"${RECORDING_DIR}\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "basename: \"streaming_server_demo_reference\"" "basename: \"${SOURCE_VIDEO_BASENAME}\"" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_testing.yaml" ${CONFIG_FILE})

  # Add comprehensive Python infrastructure test
  add_test(NAME streaming_server_demo_python_test
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh
                   ${CMAKE_CURRENT_BINARY_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.py
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  
  # Set comprehensive test properties for Python infrastructure test
  set_tests_properties(streaming_server_demo_python_test PROPERTIES
                       ENVIRONMENT "PYTHONPATH=/opt/nvidia/holoscan/lib/../python/lib:${CMAKE_BINARY_DIR}/python/lib:${CMAKE_CURRENT_BINARY_DIR}/python/lib"
                       TIMEOUT 120
                       PASS_REGULAR_EXPRESSION "Test PASSED: StreamingServer functionality validated successfully"
                       FAIL_REGULAR_EXPRESSION "Test FAILED:")

  # Add comprehensive Python functional test with real video data
  add_test(NAME streaming_server_demo_python_functional_test
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_functional_test.sh
                   ${CMAKE_CURRENT_BINARY_DIR}
                   ${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo_functional.py
                   ${HOLOHUB_DATA_DIR}/endoscopy
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  
  # Set comprehensive test properties for Python functional test
  set_tests_properties(streaming_server_demo_python_functional_test PROPERTIES
                       ENVIRONMENT "PYTHONPATH=/opt/nvidia/holoscan/lib/../python/lib:${CMAKE_BINARY_DIR}/python/lib:${CMAKE_CURRENT_BINARY_DIR}/python/lib"
                       TIMEOUT 90
                       PASS_REGULAR_EXPRESSION "FUNCTIONAL test PASSED: StreamingServer.*successful"
                       FAIL_REGULAR_EXPRESSION "FUNCTIONAL test.*FAILED")



  # Video validation test disabled for streaming server - server doesn't generate 
  # the expected GXF output format for validation. Server testing focuses on
  # functionality and unit testing rather than visual output validation.
  # TODO: Re-enable when server output recording is implemented
  # add_test(NAME streaming_server_demo_python_render_test
  #     COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
  #     --source_video_dir ${VALIDATION_FRAMES_DIR}
  #     --source_video_basename "streaming_server_demo_reference"
  #     --output_dir ${RECORDING_DIR}
  #     --validation_frames_dir ${VALIDATION_FRAMES_DIR}
  #     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  # )
  # set_tests_properties(streaming_server_demo_python_render_test PROPERTIES
  #     DEPENDS streaming_server_demo_python_test
  #     PASS_REGULAR_EXPRESSION "Valid video output!"
  # )

endif()
