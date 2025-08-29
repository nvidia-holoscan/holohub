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

# Install holoscan_ros2 Python package
add_custom_target(install_holoscan_ros2_python ALL
  COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/python/lib
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_SOURCE_DIR}/operators/holoscan_ros2/python/holoscan_ros2 ${CMAKE_BINARY_DIR}/python/lib/holoscan_ros2
  COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_SOURCE_DIR}/operators/holoscan_ros2/python/setup.py ${CMAKE_BINARY_DIR}/python/setup.py
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/python
  COMMENT "Installing holoscan_ros2 Python package"
)
