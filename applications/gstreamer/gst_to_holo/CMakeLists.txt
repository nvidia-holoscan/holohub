# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(gst_to_holo CXX)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(gst-to-holo
  gst_to_holo.cpp
)

target_include_directories(gst-to-holo
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../common
)

target_link_libraries(gst-to-holo
  PRIVATE
  holoscan::core
  holoscan::ops::holoviz
  holoscan::ops::gstreamer_bridge
)

# Install target
install(TARGETS gst-to-holo DESTINATION bin/gst_to_holo)

# Test
if(BUILD_TESTING)
  add_test(NAME gst_to_holo_test
           COMMAND gst-to-holo --count 100
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(gst_to_holo_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Application finished successfully"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
