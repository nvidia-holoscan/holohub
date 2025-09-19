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

# Copy streaming client demo application file
add_custom_target(python_streaming_client_demo_enhanced ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_client_demo.py"
  BYPRODUCTS "streaming_client_demo.py"
)

# Add Python testing - StreamingClientOp now available in Python bindings
if(BUILD_TESTING)

  set(TEST_CONFIG_DIR "${CMAKE_SOURCE_DIR}/applications/streaming_client_demo_enhanced/testing/")

  # Test 1: Python Help Test ⭐ BASIC
  add_test(NAME streaming_client_demo_enhanced_python_help_test
           COMMAND python3 streaming_client_demo.py --help
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  # To get the environment path (needed for holoscan core operators)
  find_package(holoscan 1.0 REQUIRED CONFIG PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

  set_property(TEST streaming_client_demo_enhanced_python_help_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(streaming_client_demo_enhanced_python_help_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "usage:|--help|Streaming Client Test Application"
                       FAIL_REGULAR_EXPRESSION "ImportError|ModuleNotFoundError|FATAL|CRITICAL"
                       TIMEOUT 15)

  # Test 2: Python Module Import Test ⭐ FUNCTIONAL
  add_test(NAME streaming_client_demo_enhanced_python_import_test
           COMMAND python3 -c "import sys; sys.path.append('${CMAKE_CURRENT_BINARY_DIR}'); import streaming_client_demo; print('✅ Import successful')"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_property(TEST streaming_client_demo_enhanced_python_import_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(streaming_client_demo_enhanced_python_import_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "✅ Import successful"
                       FAIL_REGULAR_EXPRESSION "ImportError|ModuleNotFoundError|FATAL|CRITICAL"
                       TIMEOUT 15)

  # Test 3: Python StreamingClientOp Binding Test ⭐ FUNCTIONAL
  add_test(NAME streaming_client_demo_enhanced_python_bindings_test
           COMMAND python3 -c "from holohub.streaming_client import StreamingClientOp; print('✅ StreamingClientOp binding available')"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_property(TEST streaming_client_demo_enhanced_python_bindings_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(streaming_client_demo_enhanced_python_bindings_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "✅ StreamingClientOp binding available"
                       FAIL_REGULAR_EXPRESSION "ImportError|ModuleNotFoundError|FATAL|CRITICAL"
                       TIMEOUT 15)

  # Test 4: Python Application Instantiation Test ⭐ FUNCTIONAL
  add_test(NAME streaming_client_demo_enhanced_python_instantiation_test
           COMMAND timeout 5s python3 -c "
import sys; sys.path.append('${CMAKE_CURRENT_BINARY_DIR}')
import streaming_client_demo
app = streaming_client_demo.StreamingClientTestApp(width=640, height=480, fps=30)
print('✅ Application instantiated successfully')
print(f'✅ Parameters: {app.width}x{app.height}@{app.fps}fps, server={app.server_ip}:{app.signaling_port}')
"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_property(TEST streaming_client_demo_enhanced_python_instantiation_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(streaming_client_demo_enhanced_python_instantiation_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "✅ Application instantiated successfully|✅ Parameters: 640x480@30fps"
                       FAIL_REGULAR_EXPRESSION "ImportError|ModuleNotFoundError|FATAL|CRITICAL"
                       TIMEOUT 10)

  # Test 5: Python Syntax Validation Test ⭐ BASIC
  add_test(NAME streaming_client_demo_enhanced_python_syntax_test
           COMMAND python3 -m py_compile streaming_client_demo.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_tests_properties(streaming_client_demo_enhanced_python_syntax_test PROPERTIES
                       FAIL_REGULAR_EXPRESSION "SyntaxError|IndentationError|FATAL|CRITICAL"
                       TIMEOUT 10)

  # Test 6: Python Configuration Parameter Test ⭐ FUNCTIONAL
  add_test(NAME streaming_client_demo_enhanced_python_parameters_test
           COMMAND timeout 3s python3 -c "
import sys; sys.path.append('${CMAKE_CURRENT_BINARY_DIR}')
import streaming_client_demo
app = streaming_client_demo.StreamingClientTestApp(
    server_ip='192.168.1.100',
    signaling_port=8080,
    width=1280,
    height=720,
    fps=60
)
print(f'✅ Custom parameters: {app.width}x{app.height}@{app.fps}fps')
print(f'✅ Network config: {app.server_ip}:{app.signaling_port}')
"
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  set_property(TEST streaming_client_demo_enhanced_python_parameters_test PROPERTY ENVIRONMENT
               "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib")

  set_tests_properties(streaming_client_demo_enhanced_python_parameters_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "✅ Custom parameters: 1280x720@60fps|✅ Network config: 192.168.1.100:8080"
                       FAIL_REGULAR_EXPRESSION "ImportError|ModuleNotFoundError|FATAL|CRITICAL"
                       TIMEOUT 8)

endif()
