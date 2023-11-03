# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  find_package(holoscan 0.5 REQUIRED CONFIG
               PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  # Add test
  # Warning: this test just check that the application starts if there is no camera
  add_test(NAME high_speed_endoscopy_test_python
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/high_speed_endoscopy.py
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_property(TEST high_speed_endoscopy_test_python PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")
  set_tests_properties(high_speed_endoscopy_test_python PROPERTIES
                       PASS_REGULAR_EXPRESSION "No EVT camera found.")
endif()
