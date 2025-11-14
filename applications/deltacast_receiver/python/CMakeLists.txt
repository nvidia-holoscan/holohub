# SPDX-FileCopyrightText: Copyright (c) 2022-2025 DELTACAST.TV. All rights reserved.
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

# To get the environment path
find_package(holoscan 1.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Add testing
if(BUILD_TESTING)

  # Add test
  add_test(NAME deltacast_receiver_python_test
           COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/deltacast_receiver.py
                   --config ${CMAKE_CURRENT_SOURCE_DIR}/deltacast_receiver.yaml
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_property(TEST deltacast_receiver_python_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(deltacast_receiver_python_test PROPERTIES
                PASS_REGULAR_EXPRESSION "Running the graph;"
                FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()

# Install application and dependencies into the install/ directory for packaging
install(
  FILES deltacast_receiver.py
  DESTINATION bin/deltacast_receiver/python
)

install(
  FILES deltacast_receiver.yaml
  DESTINATION bin/deltacast_receiver/python
)
