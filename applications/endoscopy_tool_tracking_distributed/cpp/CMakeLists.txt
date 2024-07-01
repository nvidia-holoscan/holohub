# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(endoscopy_tool_tracking_distributed CXX)

find_package(holoscan 2.1 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(PkgConfig REQUIRED)


add_executable(endoscopy_tool_tracking_distributed
  main.cpp
)

target_link_libraries(endoscopy_tool_tracking_distributed
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::video_stream_recorder
  holoscan::ops::format_converter
  holoscan::ops::holoviz
  lstm_tensor_rt_inference
  tool_tracking_postprocessor
)

target_link_libraries(endoscopy_tool_tracking_distributed PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::videomaster>)
target_link_libraries(endoscopy_tool_tracking_distributed PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::qcap_source>)
target_link_libraries(endoscopy_tool_tracking_distributed PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::vtk_renderer>)

# Download the associated dataset if needed
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  add_dependencies(endoscopy_tool_tracking_distributed endoscopy_data)
endif()

# Copy the config to the binary directory
add_custom_target(endoscopy_tool_tracking_distributed_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "endoscopy_tool_tracking.yaml"
  BYPRODUCTS "endoscopy_tool_tracking.yaml"
)
add_dependencies(endoscopy_tool_tracking_distributed endoscopy_tool_tracking_distributed_deps)

# Add testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME cpp_endoscopy_tool_tracking_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/endoscopy_tool_tracking_distributed/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "record_type: \"none\"" "record_type: \"visualizer\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "directory: \"/tmp\"" "directory: \"${RECORDING_DIR}\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "basename: \"tensor\"" "basename: \"${SOURCE_VIDEO_BASENAME}\"" CONFIG_FILE ${CONFIG_FILE})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml" ${CONFIG_FILE})

  # Add test
  add_test(NAME endoscopy_tool_tracking_distributed_cpp_test
           COMMAND endoscopy_tool_tracking_distributed ${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/endoscopy"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(endoscopy_tool_tracking_distributed_cpp_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # Add a test to check the validity of the frames
  add_test(NAME endoscopy_tool_tracking_distributed_cpp_render_test
      COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
      --source_video_dir ${RECORDING_DIR}
      --source_video_basename ${SOURCE_VIDEO_BASENAME}
      --output_dir ${RECORDING_DIR}
      --validation_frames_dir ${VALIDATION_FRAMES_DIR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(endoscopy_tool_tracking_distributed_cpp_render_test PROPERTIES
  DEPENDS endoscopy_tool_tracking_distributed_cpp_test
  PASS_REGULAR_EXPRESSION "Valid video output!"
  )

endif()