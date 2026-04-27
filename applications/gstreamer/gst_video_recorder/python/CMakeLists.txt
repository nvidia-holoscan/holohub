# SPDX-FileCopyrightText: Copyright (c) 2026, TECNALIA. All rights reserved.
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

# Copy gst_video_recorder application file
add_custom_target(python_gst_video_recorder ALL
  COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/gst_video_recorder.py"
          ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "gst_video_recorder.py"
  BYPRODUCTS "gst_video_recorder.py"
)

# Add testing
if(BUILD_TESTING)
  # To get the environment path
  find_package(holoscan 1.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(OUTPUT_FILE ${RECORDING_DIR}/python_gst_video_recorder_output.mp4)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Test 1: remove outputfile from previous tests
  add_test(
    NAME gst_video_recorder_python_cleanup_test
    COMMAND ${CMAKE_COMMAND} -E rm -f ${OUTPUT_FILE}
  )

  # Test 2: check if the pipeline runs successfully
  add_test(
    NAME gst_video_recorder_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/gst_video_recorder.py
            --source pattern
            --count 10
            -o ${OUTPUT_FILE}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_property(
    TEST gst_video_recorder_python_test
    PROPERTY ENVIRONMENT
      "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib"
  )

  set_tests_properties(gst_video_recorder_python_test PROPERTIES
    DEPENDS gst_video_recorder_python_cleanup_test
    PASS_REGULAR_EXPRESSION "EOS processed, pipeline finished cleanly"
    FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed"
  )

  # Test 3: check if the output file was created
  add_test(
    NAME gst_video_recorder_python_output_file_test
    COMMAND bash -c "test -f ${OUTPUT_FILE}"
  )

  set_tests_properties(gst_video_recorder_python_output_file_test PROPERTIES
    DEPENDS gst_video_recorder_python_test
  )

  # Test 4: check if the output file is a valid MPEG video file
  add_test(
    NAME gst_video_recorder_python_valid_mpeg_test
    COMMAND bash -c "ffprobe -v error ${OUTPUT_FILE}"
  )

  set_tests_properties(gst_video_recorder_python_valid_mpeg_test PROPERTIES
    DEPENDS gst_video_recorder_python_output_file_test
  )
endif()

# Install application into the install/ directory for packaging
install(
  FILES gst_video_recorder.py
  DESTINATION bin/gst_video_recorder/python
)
