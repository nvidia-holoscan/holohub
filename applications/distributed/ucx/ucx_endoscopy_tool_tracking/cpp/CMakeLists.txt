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
project(ucx_endoscopy_tool_tracking CXX)

find_package(holoscan 2.1 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(PkgConfig REQUIRED)


add_executable(ucx_endoscopy_tool_tracking
  main.cpp
)

target_link_libraries(ucx_endoscopy_tool_tracking
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::video_stream_recorder
  holoscan::ops::format_converter
  holoscan::ops::holoviz
  lstm_tensor_rt_inference
  tool_tracking_postprocessor
)

target_link_libraries(ucx_endoscopy_tool_tracking PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::videomaster>)
target_link_libraries(ucx_endoscopy_tool_tracking PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::qcap_source>)
target_link_libraries(ucx_endoscopy_tool_tracking PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::vtk_renderer>)

# Download the associated dataset if needed
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  add_dependencies(ucx_endoscopy_tool_tracking endoscopy_data)
endif()

# Copy the config to the binary directory
add_custom_target(ucx_endoscopy_tool_tracking_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "endoscopy_tool_tracking.yaml"
  BYPRODUCTS "endoscopy_tool_tracking.yaml"
)
add_dependencies(ucx_endoscopy_tool_tracking ucx_endoscopy_tool_tracking_deps)

# Add testing
if(BUILD_TESTING)
  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml" ${CONFIG_FILE})

  # Add test
  add_test(NAME ucx_endoscopy_tool_tracking_cpp_test_args
           COMMAND ucx_endoscopy_tool_tracking --data "${HOLOHUB_DATA_DIR}/endoscopy" --config "${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(ucx_endoscopy_tool_tracking_cpp_test_args PROPERTIES
                       FIXTURES_REQUIRED endoscopy_engine_generation_fixture
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  add_test(NAME ucx_endoscopy_tool_tracking_cpp_test_env_var
            COMMAND ucx_endoscopy_tool_tracking
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_tests_properties(ucx_endoscopy_tool_tracking_cpp_test_env_var PROPERTIES
    ENVIRONMENT "HOLOSCAN_INPUT_PATH=${HOLOHUB_DATA_DIR}/endoscopy;HOLOSCAN_CONFIG_PATH=${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml")

  set_tests_properties(ucx_endoscopy_tool_tracking_cpp_test_env_var PROPERTIES
                        FIXTURES_REQUIRED endoscopy_engine_generation_fixture
                        PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                        FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
