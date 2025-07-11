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

# Copy aja_capture application file
add_custom_target(python_aja_capture ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/aja_capture.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "aja_capture.py"
  BYPRODUCTS "aja_capture.py"
)

# Copy config file
add_custom_target(python_aja_capture_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/aja_capture.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "aja_capture.yaml"
  BYPRODUCTS "aja_capture.yaml"
)

add_dependencies(python_aja_capture python_aja_capture_yaml)

# Testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME python_aja_capture_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tests/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/aja_capture.yaml CONFIG_STRING)
  string(REPLACE "count: -1" "count: 10" CONFIG_STRING ${CONFIG_STRING})
  string(APPEND CONFIG_STRING "  enable_render_buffer_output: true\n\nrecorder:\n  directory: \"${RECORDING_DIR}\"\n  basename: \"${SOURCE_VIDEO_BASENAME}\"")
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/python_aja_capture_config.yaml)
  file(WRITE ${CONFIG_FILE} ${CONFIG_STRING})

  # Patch the current example to enable recording the rendering window
  add_custom_command(OUTPUT aja_capture_test.py
    COMMAND patch -u -o aja_capture_test.py ${CMAKE_CURRENT_SOURCE_DIR}/aja_capture.py
      ${CMAKE_CURRENT_SOURCE_DIR}/../tests/python_aja_capture.patch
  )

  add_custom_target(python_aja_capture_test ALL
    DEPENDS "aja_capture_test.py"
  )

  add_test(NAME AJA_CAPTURE_PYTHON_TEST
    COMMAND python3 aja_capture_test.py --config python_aja_capture_config.yaml
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
  )

  set_tests_properties(AJA_CAPTURE_PYTHON_TEST PROPERTIES
    DEPENDS "aja_capture_test.py"
    PASS_REGULAR_EXPRESSION "Scheduler stopped: Some entities are waiting for execution"
  )

  # Add a test to check the validity of the frames
  add_test(NAME AJA_CAPTURE_PYTHON_RENDER_TEST
      COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
      --source_video_dir ${RECORDING_DIR}
      --source_video_basename ${SOURCE_VIDEO_BASENAME}
      --output_dir ${RECORDING_DIR}
      --validation_frames_dir ${VALIDATION_FRAMES_DIR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(AJA_CAPTURE_PYTHON_RENDER_TEST PROPERTIES
    DEPENDS AJA_CAPTURE_PYTHON_TEST
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )

endif()
