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

find_package(holoscan 1.0 REQUIRED)

# Create example
add_executable(aja_capture
  aja_capture.cpp
)

target_link_libraries(aja_capture
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  holoscan::aja
)

# Copy config file
add_custom_target(aja_capture_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/aja_capture.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "aja_capture.yaml"
  BYPRODUCTS "aja_capture.yaml"
)

add_dependencies(aja_capture aja_capture_yaml)

# Testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME video_replayer_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../tests/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/aja_capture.yaml CONFIG_STRING)
  string(REPLACE "count: -1" "count: 10" CONFIG_STRING "${CONFIG_STRING}")
  set(CONFIG_FILE ${CMAKE_CURRENT_BINARY_DIR}/cpp_aja_capture_config.yaml)
  file(WRITE ${CONFIG_FILE} "${CONFIG_STRING}")

  # Patch the current example to enable recording the rendering window
  add_custom_command(OUTPUT aja_capture_test.cpp
    COMMAND patch -u -o aja_capture_test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/aja_capture.cpp
      ${CMAKE_CURRENT_SOURCE_DIR}/../tests/cpp_aja_capture.patch
  )

  # Create the test executable
  add_executable(aja_capture_test
     aja_capture_test.cpp
  )

  target_include_directories(aja_capture_test
    PRIVATE ${CMAKE_SOURCE_DIR}/tests)

  target_compile_definitions(aja_capture_test
    PRIVATE RECORD_OUTPUT RECORDING_DIR="${RECORDING_DIR}"
    PRIVATE SOURCE_VIDEO_BASENAME="${SOURCE_VIDEO_BASENAME}"
  )

  target_link_libraries(aja_capture_test
    PRIVATE
    holoscan::core
    holoscan::ops::holoviz
    holoscan::ops::video_stream_replayer
    holoscan::ops::video_stream_recorder
    holoscan::ops::format_converter
    holoscan::aja
  )

  # Add the test and make sure it runs
  add_test(NAME AJA_CAPTURE_CPP_TEST
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/aja_capture_test ${CONFIG_FILE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_tests_properties(AJA_CAPTURE_CPP_TEST PROPERTIES
    PASS_REGULAR_EXPRESSION "Scheduler stopped: Some entities are waiting for execution"
  )

  # Add a test to check the validity of the frames
  add_test(NAME AJA_CAPTURE_CPP_RENDER_TEST
      COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
      --source_video_dir ${RECORDING_DIR}
      --source_video_basename ${SOURCE_VIDEO_BASENAME}
      --output_dir ${RECORDING_DIR}
      --validation_frames_dir ${VALIDATION_FRAMES_DIR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(AJA_CAPTURE_CPP_RENDER_TEST PROPERTIES
  DEPENDS AJA_CAPTURE_CPP_TEST
  PASS_REGULAR_EXPRESSION "Valid video output!"
  )

endif()
