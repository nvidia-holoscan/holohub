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

cmake_minimum_required(VERSION 3.20)
project(video_streaming LANGUAGES NONE)

# Enable testing for video streaming applications
include(CTest)

add_holohub_application(video_streaming_client DEPENDS OPERATORS streaming_client)
add_holohub_application(video_streaming_server DEPENDS OPERATORS streaming_server)

# Testing infrastructure
if(BUILD_TESTING)
  # Video Streaming Client Tests
  add_test(NAME video_streaming_client_cpp_test
           COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_client/cpp/run_test.sh"
                   "$<TARGET_FILE:streaming_client_demo>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_client/testing/video_streaming_client_testing.yaml"
                   "${HOLOHUB_DATA_DIR}/endoscopy"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  
  set_tests_properties(video_streaming_client_cpp_test PROPERTIES
                       TIMEOUT 90
                       PASS_REGULAR_EXPRESSION "Test PASSED: C\\+\\+ StreamingClient.*test successful"
                       FAIL_REGULAR_EXPRESSION "Test FAILED:")

  add_test(NAME video_streaming_client_python_test
           COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_client/python/run_test.sh"
                   "${CMAKE_BINARY_DIR}/applications/video_streaming/video_streaming_client/python"
                   "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_client/python/streaming_client_demo.py"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_tests_properties(video_streaming_client_python_test PROPERTIES
                       ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
                       TIMEOUT 120
                       PASS_REGULAR_EXPRESSION "Test PASSED: StreamingClient functionality validated"
                       FAIL_REGULAR_EXPRESSION "Test FAILED:")

  # Video Streaming Server Tests
  add_test(NAME video_streaming_server_cpp_test
           COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_server/cpp/run_test.sh"
                   "$<TARGET_FILE:streaming_server_demo>"
                   "${CMAKE_CURRENT_SOURCE_DIR}/video_streaming_server/testing/video_streaming_server_testing.yaml"
                   "${HOLOHUB_DATA_DIR}/endoscopy"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_tests_properties(video_streaming_server_cpp_test PROPERTIES
                       TIMEOUT 120
                       PASS_REGULAR_EXPRESSION "Test PASSED: C\\+\\+ StreamingServer.*test successful"
                       FAIL_REGULAR_EXPRESSION "Test FAILED:")

endif()
