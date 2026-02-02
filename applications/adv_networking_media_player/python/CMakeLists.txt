# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Copy adv_networking_media_player application file
add_custom_target(python_adv_networking_media_player ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/adv_networking_media_player.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "adv_networking_media_player.py"
  BYPRODUCTS "adv_networking_media_player.py"
)

# Copy config file
add_custom_target(python_adv_networking_media_player_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_media_player.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "../adv_networking_media_player.yaml"
  BYPRODUCTS "adv_networking_media_player.yaml"
)

add_dependencies(python_adv_networking_media_player python_adv_networking_media_player_yaml)
