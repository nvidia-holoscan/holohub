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

  # Add test - using streaming client without bypass (connection failures are expected in test environment)
  add_test(NAME streaming_client_demo_python_test
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.py
                   --no-bypass-server
                   --data-path ${HOLOHUB_DATA_DIR}/endoscopy
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_property(TEST streaming_client_demo_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(streaming_client_demo_python_test PROPERTIES
                DEPENDS python_streaming_client_demo_deps
                PASS_REGULAR_EXPRESSION "Starting in STREAMING MODE|StreamingClient|connection|timeout|Application has finished running."
                FAIL_REGULAR_EXPRESSION "Fatal|FATAL|Segmentation fault|segfault")

  # Add a test to check the validity of the frames
  add_test(NAME streaming_client_demo_python_render_test
    COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
    --source_video_dir ${RECORDING_DIR}
    --source_video_basename ${SOURCE_VIDEO_BASENAME}
    --output_dir ${RECORDING_DIR}
    --validation_frames_dir ${VALIDATION_FRAMES_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(streaming_client_demo_python_render_test PROPERTIES
    DEPENDS streaming_client_demo_python_test
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )
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