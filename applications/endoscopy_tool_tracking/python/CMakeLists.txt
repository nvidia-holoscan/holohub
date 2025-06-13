# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copy endoscopy_tool_tracking application file
add_custom_target(python_endoscopy_tool_tracking ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "endoscopy_tool_tracking.py"
  BYPRODUCTS "endoscopy_tool_tracking.py"
)

# Copy config files
add_custom_target(python_endoscopy_tool_tracking_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking_aja_overlay.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "endoscopy_tool_tracking.yaml" "endoscopy_tool_tracking_aja_overlay.yaml"
  BYPRODUCTS "endoscopy_tool_tracking.yaml" "endoscopy_tool_tracking_aja_overlay.yaml"
)

add_dependencies(python_endoscopy_tool_tracking python_endoscopy_tool_tracking_yaml)

# Add testing
if(BUILD_TESTING)
  # To get the environment path
  find_package(holoscan 1.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME python_endoscopy_tool_tracking_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/endoscopy_tool_tracking/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "directory: \"/tmp\"" "directory: \"${RECORDING_DIR}\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "basename: \"tensor\"" "basename: \"${SOURCE_VIDEO_BASENAME}\"" CONFIG_FILE ${CONFIG_FILE})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml" ${CONFIG_FILE})

  # Add test
  add_test(NAME endoscopy_tool_tracking_python_test
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.py
                   --config ${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml
                   --data ${HOLOHUB_DATA_DIR}/endoscopy
                   --record_type visualizer
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_property(TEST endoscopy_tool_tracking_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(endoscopy_tool_tracking_python_test PROPERTIES
                DEPENDS python_endoscopy_tool_tracking_yaml
                PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # Add a test to check the validity of the frames
  add_test(NAME endoscopy_tool_tracking_python_render_test
    COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
    --source_video_dir ${RECORDING_DIR}
    --source_video_basename ${SOURCE_VIDEO_BASENAME}
    --output_dir ${RECORDING_DIR}
    --validation_frames_dir ${VALIDATION_FRAMES_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(endoscopy_tool_tracking_python_render_test PROPERTIES
    DEPENDS endoscopy_tool_tracking_python_test
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )
endif()

# Install application and dependencies into the install/ directory for packaging
install(
  FILES endoscopy_tool_tracking.py
  DESTINATION bin/endoscopy_tool_tracking/python
)

install(
  FILES endoscopy_tool_tracking.yaml
  DESTINATION bin/endoscopy_tool_tracking/python
)

install(
  FILES endoscopy_tool_tracking_aja_overlay.yaml
  DESTINATION bin/endoscopy_tool_tracking/python
)
