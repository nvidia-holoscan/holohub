# SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(orsi_in_out_body CXX)

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(orsi_in_out_body
    main.cpp
)

target_link_libraries(orsi_in_out_body
   PRIVATE
   holoscan::core
   # Holoscan SDK operators
   holoscan::ops::video_stream_replayer
   holoscan::ops::inference
   # Orsi Holohub operators
   holoscan::orsi::format_converter
   holoscan::orsi::segmentation_postprocessor
   holoscan::orsi::segmentation_preprocessor
   holoscan::orsi::visualizer
   orsi_app_lib
)

set(VIDEOMASTER_OPERATOR "")
if(ORSI_VIDEO_MASTER_SDK_FOUND)
    target_compile_definitions(orsi_in_out_body PRIVATE -DUSE_VIDEOMASTER)
    target_link_libraries(orsi_in_out_body PRIVATE holoscan::videomaster)
endif()

if(HOLOHUB_DOWNLOAD_DATASETS)
  add_dependencies(orsi_in_out_body orsi_data)
endif()

# Copy config file to the binary directory
add_custom_target(orsi_in_out_body_yaml
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/orsi_in_out_body.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/orsi_in_out_body.yaml"
)
add_dependencies(orsi_in_out_body orsi_in_out_body_yaml)

# Add testing (only if there is a display)
if(BUILD_TESTING AND DEFINED ENV{DISPLAY})
  # Add test
  # Configure the yaml file to only play 10 frames
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/orsi_in_out_body.yaml" CONFIG_FILE)
  string(REGEX REPLACE "^source:[^\n]*" "^source: replayer" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/orsi_in_out_body_testing.yaml" ${CONFIG_FILE})

  add_test(NAME orsi_in_out_body_cpp_test
           COMMAND orsi_in_out_body
                   ${CMAKE_CURRENT_BINARY_DIR}/orsi_in_out_body_testing.yaml
                   --data ${HOLOHUB_DATA_DIR}/orsi
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_tests_properties(orsi_in_out_body_cpp_test PROPERTIES
                PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
