# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copy streaming server demo application file
add_custom_target(python_streaming_server_demo_enhanced ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_server_demo.py"
  BYPRODUCTS "streaming_server_demo.py"
)

# Add testing
if(BUILD_TESTING)

  # To get the environment path (needed for holoscan core operators)
  find_package(holoscan 1.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  set(VALIDATION_FRAMES_DIR "${CMAKE_SOURCE_DIR}/applications/streaming_server_demo_enhanced/testing/")

  # Test 1: Python Help Test
  add_test(NAME streaming_server_demo_enhanced_python_help_test
           COMMAND python3 streaming_server_demo.py --help
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_property(TEST streaming_server_demo_enhanced_python_help_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(streaming_server_demo_enhanced_python_help_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "usage:|--help|streaming"
                       FAIL_REGULAR_EXPRESSION "ImportError|ModuleNotFoundError|FATAL"
                       TIMEOUT 15)

  # Test 2: Python Module Import Test
  add_test(NAME streaming_server_demo_enhanced_python_import_test
           COMMAND python3 -c "import sys; sys.path.append('${CMAKE_CURRENT_BINARY_DIR}'); import streaming_server_demo; print('Import successful')"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_property(TEST streaming_server_demo_enhanced_python_import_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(streaming_server_demo_enhanced_python_import_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Import successful"
                       FAIL_REGULAR_EXPRESSION "ImportError|ModuleNotFoundError|FATAL"
                       TIMEOUT 15)

  # Test 3: Python Configuration Test
  add_test(NAME streaming_server_demo_enhanced_python_config_test
           COMMAND python3 streaming_server_demo.py --help
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_property(TEST streaming_server_demo_enhanced_python_config_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(streaming_server_demo_enhanced_python_config_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "usage:|--help|--config"
                       FAIL_REGULAR_EXPRESSION "ImportError|ModuleNotFoundError|FATAL"
                       TIMEOUT 15)

  # Test 4: Python Syntax Validation Test
  add_test(NAME streaming_server_demo_enhanced_python_syntax_test
           COMMAND python3 -m py_compile streaming_server_demo.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_property(TEST streaming_server_demo_enhanced_python_syntax_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(streaming_server_demo_enhanced_python_syntax_test PROPERTIES
                       FAIL_REGULAR_EXPRESSION "SyntaxError|IndentationError|FATAL"
                       TIMEOUT 10)

endif()
