# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# target does nothing but be created if this folder is reached
add_custom_target(daqiri_socket_ping_python ALL)

# Copy config file
message(STATUS "Copying configuration files to ${CMAKE_CURRENT_BINARY_DIR}")
add_custom_target(daqiri_socket_ping_yaml_python
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../daqiri_socket_ping.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../daqiri_socket_ping.yaml"
)

add_dependencies(daqiri_socket_ping_python daqiri_socket_ping_yaml_python)

install(
  FILES daqiri_socket_ping.py
  DESTINATION examples/daqiri_socket_ping
  COMPONENT daqiri_socket_ping-py
)
