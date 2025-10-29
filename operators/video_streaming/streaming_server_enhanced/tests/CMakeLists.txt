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

# C++ Unit Tests for Streaming Server Operators

if(NOT BUILD_TESTING)
  return()
endif()

# Use FetchContent to get GTest if not found
find_package(GTest QUIET)

if(NOT GTest_FOUND)
  message(STATUS "GTest not found, fetching from GitHub...")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE
  )
  # For Windows: Prevent overriding the parent project's compiler/linker settings
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # Create aliases to match find_package targets
  if(NOT TARGET GTest::gtest)
    add_library(GTest::gtest ALIAS gtest)
  endif()
  if(NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main ALIAS gtest_main)
  endif()
endif()

# Add test executable
add_executable(test_streaming_server_ops
  test_streaming_server_ops.cpp
)

# Link against the streaming server libraries and GTest
target_link_libraries(test_streaming_server_ops
  PRIVATE
    streaming_server_enhanced
    holoscan::core
    GTest::gtest
    GTest::gtest_main
)

# Add test to CTest
add_test(NAME streaming_server_ops_unit_tests
         COMMAND test_streaming_server_ops)

# Set test properties
set_tests_properties(streaming_server_ops_unit_tests PROPERTIES
  TIMEOUT 60
  LABELS "unit;streaming_server"
)

# Set output directory
set_target_properties(test_streaming_server_ops PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators/video_streaming/streaming_server_enhanced/tests"
)

