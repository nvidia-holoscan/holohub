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

project(pva_radar_graphics LANGUAGES CXX)

add_library(pva_radar_graphics_common SHARED
  pva_radar_graphics.cpp
)

target_include_directories(pva_radar_graphics_common
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_library(holoscan::pva_radar_graphics ALIAS pva_radar_graphics_common)

target_link_libraries(pva_radar_graphics_common
  PUBLIC
    holoscan::core
    nvcv_types
    pva_operator
    cupva_host
)

set_target_properties(pva_radar_graphics_common PROPERTIES
  OUTPUT_NAME "holoscan_pva_radar_graphics"
  EXPORT_NAME "pva_radar_graphics"
)

install(
  TARGETS pva_radar_graphics_common
  EXPORT holoscan-pva-radar-graphics-targets
  COMPONENT pva_radar_graphics-cpp
)
install(
  FILES pva_radar_graphics.hpp
  DESTINATION include
  COMPONENT pva_radar_graphics-cpp
)
