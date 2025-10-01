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
project(video_streaming_demo_enhanced LANGUAGES NONE)

add_holohub_application(video_streaming_demo_client DEPENDS OPERATORS video_streaming)
add_holohub_application(video_streaming_demo_server DEPENDS OPERATORS video_streaming)

# Build client and server subdirectories
add_subdirectory(video_streaming_demo_client)
add_subdirectory(video_streaming_demo_server)

# Add integration testing
if(BUILD_TESTING)
  # Copy integration test script
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/integration_test.sh"
    "${CMAKE_BINARY_DIR}/integration_test.sh"
    COPYONLY
  )
  
  # Make script executable
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/integration_test.sh"
       DESTINATION "${CMAKE_BINARY_DIR}"
       FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
  
  # Add integration test for video streaming demo
  add_test(NAME video_streaming_integration_test
           COMMAND bash "${CMAKE_BINARY_DIR}/integration_test.sh"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  
  set_tests_properties(video_streaming_integration_test PROPERTIES
                       TIMEOUT 120
                       PASS_REGULAR_EXPRESSION "Integration test PASSED"
                       FAIL_REGULAR_EXPRESSION "Integration test FAILED")
endif()