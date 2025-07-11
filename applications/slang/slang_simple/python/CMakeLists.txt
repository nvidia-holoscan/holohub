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

if(BUILD_TESTING)
  add_test(NAME slang_simple_python_test
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/slang_simple.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
  set_property(TEST slang_simple_python_test PROPERTY ENVIRONMENT
           "PYTHONPATH=$ENV{PYTHONPATH}:${CMAKE_BINARY_DIR}/python/lib")
  set_tests_properties(slang_simple_python_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Received value: 20"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed;error")
endif()
