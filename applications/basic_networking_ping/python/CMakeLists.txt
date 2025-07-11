# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
# to set dependencies on adequate operators
add_custom_target(basic_networking_ping_python ALL
  DEPENDS basic_network_python
)

# Copy config file
message(STATUS "Copying configuration files to ${CMAKE_CURRENT_BINARY_DIR}")
add_custom_target(basic_networking_ping_rx_yaml_python
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../basic_networking_ping_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../basic_networking_ping_rx.yaml"
)
add_custom_target(basic_networking_ping_tx_yaml_python
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../basic_networking_ping_tx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../basic_networking_ping_tx.yaml"
)

add_dependencies(basic_networking_ping_python basic_networking_ping_rx_yaml_python)
add_dependencies(basic_networking_ping_python basic_networking_ping_tx_yaml_python)

install(
  FILES basic_networking_ping.py
  DESTINATION examples/basic_networking_ping
  COMPONENT basic_networking_ping-py
)
