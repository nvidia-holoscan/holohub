# SPDX-FileCopyrightText: Copyright (c) 2025-2026, Holoscan Team
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

project(ucxx_endoscopy_tool_tracking)

find_package(holoscan 3.9.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")


add_executable(ucxx_endoscopy_tool_tracking
    main.cpp
    publisher.cpp
    subscriber_holoviz.cpp
    overlay.cpp
)

target_include_directories(ucxx_endoscopy_tool_tracking
    PRIVATE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
)

target_link_libraries(ucxx_endoscopy_tool_tracking
    PRIVATE
    holoscan::core
    holoscan::ops::video_stream_replayer
    holoscan::ops::video_stream_recorder
    holoscan::ops::format_converter
    holoscan::ops::holoviz
    holoscan::ops::ucxx
    lstm_tensor_rt_inference
    tool_tracking_postprocessor
)

# Copy configuration file
add_custom_target(ucxx_endoscopy_tool_tracking_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ucxx_endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "ucxx_endoscopy_tool_tracking.yaml"
)

add_dependencies(ucxx_endoscopy_tool_tracking ucxx_endoscopy_tool_tracking_yaml)

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the endoscopy sample data
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL nvidia/clara-holoscan/holoscan_endoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 480
    GXF_ENTITIES_WIDTH 854
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )
  add_dependencies(ucxx_endoscopy_tool_tracking endoscopy_data)
endif()


if(OFF)  # TODO: Implement frame count cutoff in subscriber application for testing

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME cpp_ucxx_endoscopy_tool_tracking_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/endoscopy_tool_tracking/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/ucxx_endoscopy_tool_tracking.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "headless: false" "headless: true" CONFIG_FILE ${CONFIG_FILE})

  # Enable subscriber-side recording for frame validation
  string(REPLACE "record_type: \"none\"" "record_type: \"subscriber\"" CONFIG_FILE ${CONFIG_FILE})

  # Update recorder paths for testing
  string(REPLACE "directory: \"/tmp\"" "directory: \"${RECORDING_DIR}\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "basename: \"ucxx_subscriber_received\"" "basename: \"${SOURCE_VIDEO_BASENAME}\"" CONFIG_FILE ${CONFIG_FILE})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ucxx_endoscopy_tool_tracking_testing.yaml" ${CONFIG_FILE})

  # Add test that runs both publisher and subscriber
  # Note: The test runs the server in the background and connects a client to it
  add_test(NAME ucxx_endoscopy_tool_tracking_cpp_test
           COMMAND bash -c "\
             ${CMAKE_CURRENT_BINARY_DIR}/ucxx_endoscopy_tool_tracking \
               --mode publisher \
               --config ${CMAKE_CURRENT_BINARY_DIR}/ucxx_endoscopy_tool_tracking_testing.yaml \
               --data ${HOLOHUB_DATA_DIR}/endoscopy \
               --hostname 127.0.0.1 \
               --port 50008 & \
             PUBLISHER_PID=$!; \
             sleep 2; \
             ${CMAKE_CURRENT_BINARY_DIR}/ucxx_endoscopy_tool_tracking \
               --mode subscriber \
               --config ${CMAKE_CURRENT_BINARY_DIR}/ucxx_endoscopy_tool_tracking_testing.yaml \
               --hostname 127.0.0.1 \
               --port 50008 & \
             SUBSCRIBER_PID=$!; \
             wait $SUBSCRIBER_PID; \
             SUBSCRIBER_EXIT=$?; \
             wait $PUBLISHER_PID; \
             PUBLISHER_EXIT=$?; \
             if [ $SUBSCRIBER_EXIT -ne 0 ] || [ $PUBLISHER_EXIT -ne 0 ]; then \
               exit 1; \
             fi; \
             exit 0"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_tests_properties(ucxx_endoscopy_tool_tracking_cpp_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # Set a timeout for the test (60 seconds should be enough for 10 frames)
  set_tests_properties(ucxx_endoscopy_tool_tracking_cpp_test PROPERTIES TIMEOUT 60)

  # Add test to check the validity of frames received by the subscriber
  # This validates the end-to-end distributed pipeline
  add_test(NAME ucxx_endoscopy_tool_tracking_cpp_render_test
      COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
      --source_video_dir ${RECORDING_DIR}
      --source_video_basename ${SOURCE_VIDEO_BASENAME}
      --output_dir ${RECORDING_DIR}
      --validation_frames_dir ${VALIDATION_FRAMES_DIR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(ucxx_endoscopy_tool_tracking_cpp_render_test PROPERTIES
    DEPENDS ucxx_endoscopy_tool_tracking_cpp_test
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )

endif()

