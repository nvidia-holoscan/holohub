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

cmake_minimum_required(VERSION 3.20)
project(multiai_ultrasound CXX)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(multiai_ultrasound
    main.cpp
)

target_link_libraries(multiai_ultrasound
    PRIVATE
    holoscan::core
    holoscan::ops::video_stream_recorder
    holoscan::ops::video_stream_replayer
    holoscan::ops::format_converter
    holoscan::ops::holoviz
    holoscan::ops::inference
    holoscan::ops::inference_processor
    holoscan::aja
    visualizer_icardio
)

# Download the associated dataset if needed
if(HOLOHUB_DOWNLOAD_DATASETS)
  add_dependencies(multiai_ultrasound multiai_ultrasound_data)
endif()

# Copy config file
add_custom_target(multiai_ultrasound_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "multiai_ultrasound.yaml"
    BYPRODUCTS "multiai_ultrasound.yaml"
)
add_dependencies(multiai_ultrasound multiai_ultrasound_yaml)

add_custom_target(mgpu_multiai_ultrasound_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/mgpu_multiai_ultrasound.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "mgpu_multiai_ultrasound.yaml"
    BYPRODUCTS "mgpu_multiai_ultrasound.yaml"
)
add_dependencies(multiai_ultrasound mgpu_multiai_ultrasound_yaml)

# Add testing
if(BUILD_TESTING)
  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME cpp_multiai_ultrasound_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/multiai_ultrasound/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/multiai_ultrasound.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REGEX REPLACE "source:[^\n]*" "source: replayer" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "record_type: \"none\"" "record_type: \"visualizer\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "directory: \"/tmp\"" "directory: \"${RECORDING_DIR}\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "basename: \"tensor\"" "basename: \"${SOURCE_VIDEO_BASENAME}\"" CONFIG_FILE ${CONFIG_FILE})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/multiai_ultrasound_testing.yaml" ${CONFIG_FILE})

  # Add test
  add_test(NAME multiai_ultrasound_cpp_test
           COMMAND multiai_ultrasound ${CMAKE_CURRENT_BINARY_DIR}/multiai_ultrasound_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/multiai_ultrasound"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(multiai_ultrasound_cpp_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # Add a test to check the validity of the frames
  add_test(NAME multiai_ultrasound_cpp_render_test
    COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
    --source_video_dir ${RECORDING_DIR}
    --source_video_basename ${SOURCE_VIDEO_BASENAME}
    --output_dir ${RECORDING_DIR}
    --validation_frames_dir ${VALIDATION_FRAMES_DIR}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(multiai_ultrasound_cpp_render_test PROPERTIES
    DEPENDS multiai_ultrasound_cpp_test
    PASS_REGULAR_EXPRESSION "Valid video output!"
  )

endif()
