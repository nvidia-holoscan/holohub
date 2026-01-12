# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# C++ Unit Tests for VideoStreamingClientOp

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
add_executable(test_video_streaming_client_op
  test_video_streaming_client_op.cpp
)

# Link against the video streaming client library and GTest
target_link_libraries(test_video_streaming_client_op
  PRIVATE
    video_streaming_client
    holoscan::core
    GTest::gtest
    GTest::gtest_main
)

# Include directories
target_include_directories(test_video_streaming_client_op
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
)

# Add test to CTest
add_test(NAME video_streaming_client_op_unit_tests
         COMMAND test_video_streaming_client_op)

# Set test properties
set_tests_properties(video_streaming_client_op_unit_tests PROPERTIES
  TIMEOUT 60
  LABELS "unit;video_streaming_client"
)

# Set output directory and RPATH
# Priority: System OpenSSL 3.4 > Holoscan libs (UCX) > system CUDA > build dirs > system libs
set_target_properties(test_video_streaming_client_op PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/operators/video_streaming/video_streaming_client/tests"
  SKIP_BUILD_RPATH FALSE
  BUILD_WITH_INSTALL_RPATH FALSE
  BUILD_RPATH "/usr/local/ssl/lib64:/opt/nvidia/holoscan/lib:/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:${CMAKE_BINARY_DIR}/lib:${CMAKE_BINARY_DIR}/operators/video_streaming/video_streaming_client:/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
  INSTALL_RPATH "/usr/local/ssl/lib64:/opt/nvidia/holoscan/lib:/usr/local/cuda/lib64:/usr/local/cuda-12/lib64:$ORIGIN:$ORIGIN/../lib:$ORIGIN/../../lib:/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

