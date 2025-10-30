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

# Copy streaming_server_demo application file

# Copy streaming_server_demo application file to both locations
add_custom_target(python_streaming_server_demo ALL
  COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.py"
          "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo.py"
  COMMAND ${CMAKE_COMMAND} -E copy
          "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.py"
          "${CMAKE_BINARY_DIR}/streaming_server_demo.py"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.py"
  BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo.py"
             "${CMAKE_BINARY_DIR}/streaming_server_demo.py"
)

# Copy YAML configuration files if they exist
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml")
  configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml"
    "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo.yaml"
    COPYONLY
  )
endif()

# Install Python application
install(
  FILES streaming_server_demo.py
  DESTINATION bin/video_streaming/python
  PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE
)

# Install configuration files
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml")
  install(
    FILES streaming_server_demo.yaml
    DESTINATION bin/video_streaming/python
  )
endif()
