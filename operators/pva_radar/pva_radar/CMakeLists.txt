# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
cmake_minimum_required(VERSION 3.20)

project(pva_radar LANGUAGES CXX)

set(RADAR_PIPELINE_SRC_DIR "${PVA_SOLUTIONS_SRC_ROOT}/pipelines/radar")

add_library(pva_radar_common SHARED
  impl/radar_pipeline_utils.cpp
  impl/radar_pipeline_pva.cpp
  impl/radar_pipeline_tensors.cpp
  impl/PVARadarPipeline.cpp
  pva_radar.cpp
)

target_include_directories(pva_radar_common
  PRIVATE
    ${RADAR_PIPELINE_SRC_DIR}
    ${RADAR_PIPELINE_SRC_DIR}/operators/doa/ref
    ${PVA_SOLUTIONS_SRC_ROOT}/samples/range_fft/ref
    ${PVA_SOLUTIONS_SRC_ROOT}/samples/doppler_fft/ref
  INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

add_library(holoscan::pva_radar ALIAS pva_radar_common)

target_link_libraries(pva_radar_common
  PUBLIC
    holoscan::core
    radar_operators
    pva_operator
    cupva_host
    nvcv_types
)

set_target_properties(pva_radar_common PROPERTIES
  OUTPUT_NAME "holoscan_pva_radar"
  EXPORT_NAME "pva_radar"
)

install(
  TARGETS pva_radar_common
  EXPORT holoscan-pva-radar-targets
  COMPONENT pva_radar-cpp
)
install(
  FILES pva_radar.hpp
  DESTINATION include
  COMPONENT pva_radar-cpp
)
