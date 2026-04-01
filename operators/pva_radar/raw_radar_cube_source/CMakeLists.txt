# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

project(raw_radar_cube_source LANGUAGES CXX)

add_library(raw_radar_cube_source_common SHARED
  ${PVA_SOLUTIONS_SRC_ROOT}/pipelines/radar/utils/radar_file_parser.cpp
  raw_radar_cube_source.cpp
)

target_include_directories(raw_radar_cube_source_common
  PRIVATE
    ${PVA_SOLUTIONS_SRC_ROOT}/pipelines/radar/utils
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_library(holoscan::raw_radar_cube_source ALIAS raw_radar_cube_source_common)

target_link_libraries(raw_radar_cube_source_common
  PUBLIC
    holoscan::core
    nvcv_types
    pva_operator
    cupva_host
)

set_target_properties(raw_radar_cube_source_common PROPERTIES
  OUTPUT_NAME "holoscan_raw_radar_cube_source"
  EXPORT_NAME "raw_radar_cube_source"
)

install(
  TARGETS raw_radar_cube_source_common
  EXPORT holoscan-raw-radar-cube-source-targets
  COMPONENT raw_radar_cube_source-cpp
)
install(
  FILES raw_radar_cube_source.hpp
  DESTINATION include
  COMPONENT raw_radar_cube_source-cpp
)
