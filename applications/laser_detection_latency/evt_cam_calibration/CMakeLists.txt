# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.24)
project(evt_cam_calibration)

find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_custom_target(evt_cam_calibration_deps ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../images/apriltag-calibration.png" "${CMAKE_CURRENT_BINARY_DIR}/apriltag-calibration.png"
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../images/apriltag-calibration.png"
  BYPRODUCTS "apriltag-calibration.png"
)

add_library(evt_cam_calibration INTERFACE)
add_dependencies(evt_cam_calibration evt_cam_calibration_deps)
