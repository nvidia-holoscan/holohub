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
   qcap_source
)

target_include_directories(ultrasound_segmentation
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../operators/qcap_source
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
  # Configure the yaml file to only play 10 frames
  file(READ ${CMAKE_CURRENT_SOURCE_DIR}/ultrasound_segmentation.yaml CONFIG_FILE)
  string(REGEX REPLACE "source:[^\n]*" "source: replayer" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ultrasound_segmentation_testing.yaml ${CONFIG_FILE})

  # Add test
  add_test(NAME ultrasound_segmentation_test
           COMMAND ultrasound_segmentation ${CMAKE_CURRENT_BINARY_DIR}/ultrasound_segmentation_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/ultrasound_segmentation"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  # Check the terminal output
  set_tests_properties(ultrasound_segmentation_test PROPERTIES
                       ENVIRONMENT "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}:${CMAKE_BINARY_DIR}/gxf_extensions"
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
