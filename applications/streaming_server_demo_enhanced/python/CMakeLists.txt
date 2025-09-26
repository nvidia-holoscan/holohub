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

# No application tests - all testing is done at operator level
# For comprehensive testing including Python bindings, use:
# cmake -DBUILD_TESTING=ON -DHOLOHUB_BUILD_PYTHON=ON -DHOLOHUB_BUILD_OPERATORS="streaming_server_enhanced"
# Then: ctest -L streaming_server
