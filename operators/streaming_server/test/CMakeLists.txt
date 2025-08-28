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

# Set up GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Create the test executable
add_executable(streaming_server_operator_test
  test_streaming_server_op.cpp
)

# Link with required libraries
target_link_libraries(streaming_server_operator_test
  PRIVATE
  holoscan::core
  holoscan::streaming_server
  GTest::gtest_main
  GTest::gmock
)

# Include the operator source directory for access to headers
target_include_directories(streaming_server_operator_test
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Add the test to CTest
include(GoogleTest)
gtest_discover_tests(streaming_server_operator_test)

# Alternative: Add as a single test
add_test(NAME streaming_server_operator_test COMMAND streaming_server_operator_test)
