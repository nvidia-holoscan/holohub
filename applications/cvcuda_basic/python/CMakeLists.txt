# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  find_package(holoscan 3.4 REQUIRED CONFIG
               PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  # Add test
  add_test(NAME cvcuda_basic_python_test
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/cvcuda_basic.py
                   --data ${HOLOHUB_DATA_DIR}/endoscopy --count 10
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_property(TEST cvcuda_basic_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(cvcuda_basic_python_test
                PROPERTIES PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking.;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
