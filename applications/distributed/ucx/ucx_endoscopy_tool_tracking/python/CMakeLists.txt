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

# Add testing
if(BUILD_TESTING)
  # To get the environment path
  find_package(holoscan 2.1 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml" ${CONFIG_FILE})

  # Add test
  add_test(NAME ucx_endoscopy_tool_tracking_python_test_args
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.py --data "${HOLOHUB_DATA_DIR}/endoscopy" --config "${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_tests_properties(ucx_endoscopy_tool_tracking_python_test_args PROPERTIES
                       ENVIRONMENT "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(ucx_endoscopy_tool_tracking_python_test_args PROPERTIES
                       FIXTURES_REQUIRED endoscopy_engine_generation_fixture
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  add_test(NAME ucx_endoscopy_tool_tracking_python_test_env_vars
          COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.py --data "${HOLOHUB_DATA_DIR}/endoscopy" --config "${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml"
          WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_tests_properties(ucx_endoscopy_tool_tracking_python_test_env_vars PROPERTIES
                       ENVIRONMENT "HOLOSCAN_INPUT_PATH=${HOLOHUB_DATA_DIR}/endoscopy;HOLOSCAN_CONFIG_PATH=${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml;PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(ucx_endoscopy_tool_tracking_python_test_env_vars PROPERTIES
                       FIXTURES_REQUIRED endoscopy_engine_generation_fixture
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

endif()
