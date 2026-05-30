# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(point_cloud_from_depth_demo NONE)

find_package(holoscan 3.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

if(BUILD_TESTING)
  add_test(NAME point_cloud_from_depth_demo_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/point_cloud_from_depth_demo.py --frames 10
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_property(TEST point_cloud_from_depth_demo_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(point_cloud_from_depth_demo_python_test
    PROPERTIES
      PASS_REGULAR_EXPRESSION "valid="
      FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
