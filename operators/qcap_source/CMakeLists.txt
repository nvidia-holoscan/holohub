# SPDX-FileCopyrightText: Copyright (c) 2022-2023 YUAN High-Tech Development Co., Ltd. All rights reserved.
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
project(qccap_source)

find_package(holoscan 0.5 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_library(qcap_source SHARED
  qcap_source.hpp
  qcap_source.cpp
  )
add_library(holoscan::qcap_source ALIAS qcap_source)
target_include_directories(qcap_source INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qcap_source holoscan::core)
target_link_libraries(qcap_source gxf_qcap_source)

if(HOLOHUB_BUILD_PYTHON)
    add_subdirectory(python)
endif()
