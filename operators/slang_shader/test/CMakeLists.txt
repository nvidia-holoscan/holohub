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

# Fetch GoogleTest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

# Create test executable
add_executable(slang_shader_test
  test_slang_shader_op.cpp
)

# Link libraries
target_link_libraries(slang_shader_test
  holoscan::core
  GXF::multimedia
  holoscan::ops::slang_shader
  GTest::gtest_main
  GTest::gmock
)

# Include directories
target_include_directories(slang_shader_test
  PRIVATE
    ${CMAKE_SOURCE_DIR}/operators/slang_shader
)

# Add test to CTest
add_test(NAME slang_shader_test COMMAND slang_shader_test)
