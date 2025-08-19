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

# Copy streaming_client_demo application file
add_custom_target(python_streaming_client_demo ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "streaming_client_demo.py"
  BYPRODUCTS "streaming_client_demo.py"
)

# Install application and dependencies into the install/ directory for packaging
install(
  FILES streaming_client_demo.py
  DESTINATION bin/streaming_client_demo/python
)