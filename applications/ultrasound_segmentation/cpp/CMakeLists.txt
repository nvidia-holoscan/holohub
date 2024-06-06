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

cmake_minimum_required(VERSION 3.20)
project(ultrasound_segmentation CXX)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(ultrasound_segmentation
    main.cpp
)

target_link_libraries(ultrasound_segmentation
   PRIVATE
   holoscan::core
   holoscan::ops::aja
   holoscan::ops::video_stream_replayer
   holoscan::ops::format_converter
   holoscan::ops::inference
   holoscan::ops::segmentation_postprocessor
   holoscan::ops::holoviz
)

# Download the associated dataset if needed
if(HOLOHUB_DOWNLOAD_DATASETS)
  add_dependencies(ultrasound_segmentation ultrasound_segmentation_data)
endif()

# Copy config file
add_custom_target(ultrasound_segmentation_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/ultrasound_segmentation.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "ultrasound_segmentation.yaml"
    BYPRODUCTS "ultrasound_segmentation.yaml"
)
add_dependencies(ultrasound_segmentation ultrasound_segmentation_yaml)

# Add testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME cpp_ultrasound_segmentation_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/ultrasound_segmentation/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/ultrasound_segmentation.yaml" CONFIG_FILE)
  string(REGEX REPLACE "source:[^\n]*" "source: replayer" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ultrasound_segmentation_test.yaml ${CONFIG_FILE})

    # Patch the current example to enable recording the rendering window
    add_custom_command(OUTPUT main_test.cpp
    PRE_LINK
    COMMAND patch -u -o main_test.cpp ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
        ${CMAKE_SOURCE_DIR}/applications/ultrasound_segmentation/testing/cpp_ultrasound_segmentation.patch
    COMMAND sed -i "s#RECORDING_DIR#${RECORDING_DIR}#g" main_test.cpp
    COMMAND sed -i "s#SOURCE_VIDEO_BASENAME#${SOURCE_VIDEO_BASENAME}#g" main_test.cpp
  )

  # Create the test executable
  add_executable(ultrasound_segmentation_cpp_test
    main_test.cpp
  )

  target_link_libraries(ultrasound_segmentation_cpp_test
    PRIVATE
    holoscan::core
    holoscan::ops::aja
    holoscan::ops::video_stream_replayer
    holoscan::ops::video_stream_recorder
    holoscan::ops::format_converter
    holoscan::ops::inference
    holoscan::ops::segmentation_postprocessor
    holoscan::ops::holoviz
  )

  # Add the test and make sure it runs
  add_test(NAME ultrasound_segmentation_cpp_test
    COMMAND ultrasound_segmentation_cpp_test ${CMAKE_CURRENT_BINARY_DIR}/ultrasound_segmentation_test.yaml
            --data "${HOLOHUB_DATA_DIR}/ultrasound_segmentation"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(ultrasound_segmentation_cpp_test PROPERTIES
                       ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/gxf_extensions"
                       TIMEOUT 1500
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # Add a test to check the validity of the frames
  add_test(NAME ultrasound_segmentation_cpp_render_test
    COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
    --source_video_dir ${RECORDING_DIR}
    --source_video_basename ${SOURCE_VIDEO_BASENAME}
    --output_dir ${RECORDING_DIR}
    --validation_frames_dir ${VALIDATION_FRAMES_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(ultrasound_segmentation_cpp_render_test PROPERTIES
    DEPENDS ultrasound_segmentation_cpp_test
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )

endif()

