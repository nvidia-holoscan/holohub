# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(multiai_endoscopy CXX)


# Holoscan
find_package(Holoscan REQUIRED CONFIG 
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(multiai_endoscopy
  multi_ai.cpp
)

target_link_libraries(multiai_endoscopy
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
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  add_dependencies(multiai_endoscopy endoscopy_data ssd_model_data monai_tool_seg_model_data)
endif()

# Copy config file
add_custom_target(app_config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/app_config.yaml" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "app_config.yaml"
    BYPRODUCTS "app_config.yaml"
)
add_dependencies(multiai_endoscopy app_config_yaml)

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file to only play 10 frames
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/app_config.yaml" CONFIG_FILE)
  string(REGEX REPLACE "source:[^\n]*" "source: replayer" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/app_config_testing.yaml" ${CONFIG_FILE})

  # Add test
  add_test(NAME multiai_endoscopy_cpp_test
           COMMAND multiai_endoscopy ${CMAKE_CURRENT_BINARY_DIR}/app_config_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(multiai_endoscopy_cpp_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
