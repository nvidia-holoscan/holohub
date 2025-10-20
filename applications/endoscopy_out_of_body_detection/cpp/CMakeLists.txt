# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_executable(endoscopy_out_of_body_detection
    main.cpp
)

target_link_libraries(endoscopy_out_of_body_detection
    PRIVATE
    holoscan::core
    holoscan::ops::video_stream_replayer
    holoscan::ops::format_converter
    holoscan::ops::inference
    holoscan::ops::inference_processor
    $<TARGET_NAME_IF_EXISTS:holoscan::aja>
)

# Add downloaded data as a dependency
if(HOLOHUB_DOWNLOAD_DATASETS)
    add_dependencies(endoscopy_out_of_body_detection endoscopy_out_of_body_detection_data)
endif()

# Copy config file to the build directory
add_custom_target(endoscopy_out_of_body_detection_yaml
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/../endoscopy_out_of_body_detection.yaml"
          ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../endoscopy_out_of_body_detection.yaml"
  BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/endoscopy_out_of_body_detection.yaml"
)
add_dependencies(endoscopy_out_of_body_detection endoscopy_out_of_body_detection_yaml)

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file to only play 10 frames
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../endoscopy_out_of_body_detection.yaml" CONFIG_FILE)
  string(REGEX REPLACE "source:[^\n]*" "source: replayer" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/endoscopy_out_of_body_detection_testing.yaml" ${CONFIG_FILE})

  # Add test
  add_test(NAME endoscopy_out_of_body_detection_test
           COMMAND endoscopy_out_of_body_detection
                   --config ${CMAKE_CURRENT_BINARY_DIR}/endoscopy_out_of_body_detection_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/endoscopy_out_of_body_detection"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(endoscopy_out_of_body_detection_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
