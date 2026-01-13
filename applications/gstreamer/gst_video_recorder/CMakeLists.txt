# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(gst_video_recorder CXX)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(gst-video-recorder
  ../common/pattern_generator.cpp
  gst_video_recorder.cpp
)

target_include_directories(gst-video-recorder
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(gst-video-recorder
  PRIVATE
  holoscan::core
  holoscan::ops::gstreamer_bridge
  holoscan::ops::v4l2
  holoscan::ops::format_converter
)

# Install target
install(TARGETS gst-video-recorder DESTINATION bin/gst_video_recorder)


if(BUILD_TESTING)
    # Set and create the recording directory
    set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
    file(MAKE_DIRECTORY ${RECORDING_DIR})

    # Set the output file
    set(OUTPUT_FILE ${RECORDING_DIR}/test_output.mp4)
    message(STATUS "Output test file: ${OUTPUT_FILE}")

    # Test 1: check if the pipeline runs successfully
    add_test(
        NAME gst_video_recorder_test
        COMMAND $<TARGET_FILE:gst-video-recorder> --source pattern --count 10 -o ${OUTPUT_FILE}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )

    # Set the test passing if the pipeline runs successfully
    set_tests_properties(gst_video_recorder_test PROPERTIES
        PASS_REGULAR_EXPRESSION "EOS processed, pipeline finished cleanly"
    )

    # Test 2: check if the output file was created
    add_test(
        NAME gst_video_recorder_output_file_test
        COMMAND bash -c "test -f ${OUTPUT_FILE}"
    )

    # Make the file check test depend on the pipeline test
    set_tests_properties(gst_video_recorder_output_file_test PROPERTIES
        DEPENDS gst_video_recorder_test
    )

    # Test 3: check if the output file is a valid MPEG video file
    add_test(
        NAME gst_video_recorder_valid_mpeg_test
        COMMAND bash -c "ffprobe -v error ${OUTPUT_FILE}"
    )

    # Make the validation test depend on the file check test
    set_tests_properties(gst_video_recorder_valid_mpeg_test PROPERTIES
        DEPENDS gst_video_recorder_output_file_test
    )
endif()
