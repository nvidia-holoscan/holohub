# SPDX-FileCopyrightText: Copyright (c) 2025 XRlabs. All rights reserved.
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

project(st2110_test NONE)

# This is a Python-only test application for the ST2110 source operator
# No C++ compilation required - all functionality is in Python

# Add testing
if(BUILD_TESTING)
  add_test(NAME st2110_test_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/st2110_test_app.py
                   --config ${CMAKE_CURRENT_SOURCE_DIR}/st2110_test_config.yaml
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_tests_properties(st2110_test_python_test
                PROPERTIES FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
