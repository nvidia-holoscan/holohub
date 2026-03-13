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

# This file sets up dependencies on pva-solutions and nvcv libraries and headers by looking in the
# specified source location and default install path for deb packages.

# unless we find out otherwise...
set(pva-solutions_FOUND TRUE)

if(NOT (IS_DIRECTORY "${PVA_SOLUTIONS_SRC_ROOT}"))
    message(WARNING "PVA_SOLUTIONS_SRC_ROOT is invalid or not set: ${PVA_SOLUTIONS_SRC_ROOT}")
    set(pva-solutions_FOUND FALSE)
    return()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  set(PVA_SOLUTIONS_ARCH "x86_64-linux-gnu")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  set(PVA_SOLUTIONS_ARCH "aarch64-linux-gnu")
else()
  message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(PVA_SOLUTIONS_VERSION "0.4")
set(PVA_SOLUTIONS_LIB_ROOT "/opt/nvidia/pva-solutions-${PVA_SOLUTIONS_VERSION}/lib/${PVA_SOLUTIONS_ARCH}")

# NVCV Types library (from pva-solutions deb)
find_library(NVCV_TYPES_LIB "nvcv_types_d" PATHS ${PVA_SOLUTIONS_LIB_ROOT} NO_DEFAULT_PATH)
if(NOT NVCV_TYPES_LIB)
  message(WARNING "nvcv_types_d library not found in ${PVA_SOLUTIONS_LIB_ROOT}")
  set(pva-solutions_FOUND FALSE)
  return()
endif()
add_library(nvcv_types SHARED IMPORTED)
set_target_properties(nvcv_types PROPERTIES IMPORTED_LOCATION ${NVCV_TYPES_LIB})
target_include_directories(nvcv_types INTERFACE "${PVA_SOLUTIONS_SRC_ROOT}/3rdparty/cvcuda/src/nvcv/src/include")

# PVA Operator library (from pva-solutions deb)
find_library(PVA_OPS_LIB "pva_operator" PATHS ${PVA_SOLUTIONS_LIB_ROOT} NO_DEFAULT_PATH)
if(NOT PVA_OPS_LIB)
  message(WARNING "pva_operator library not found in ${PVA_SOLUTIONS_LIB_ROOT}")
  set(pva-solutions_FOUND FALSE)
  return()
endif()
add_library(pva_operator SHARED IMPORTED)
set_target_properties(pva_operator PROPERTIES IMPORTED_LOCATION ${PVA_OPS_LIB})

target_include_directories(pva_operator INTERFACE
    "${PVA_SOLUTIONS_SRC_ROOT}/src/operator/include"
    "${PVA_SOLUTIONS_SRC_ROOT}/src/operator/priv"
)

# PVA Radar Operators library (from pva-solutions deb)
find_library(PVA_RADAR_LIB "radar_operators"
             PATHS ${PVA_SOLUTIONS_LIB_ROOT} NO_DEFAULT_PATH)
if(NOT PVA_RADAR_LIB)
  message(WARNING "radar_operators library not found in ${PVA_SOLUTIONS_LIB_ROOT}")
  set(pva-solutions_FOUND FALSE)
  return()
endif()
add_library(radar_operators SHARED IMPORTED)
set_target_properties(radar_operators PROPERTIES IMPORTED_LOCATION ${PVA_RADAR_LIB})
target_include_directories(radar_operators INTERFACE
    "${PVA_SOLUTIONS_SRC_ROOT}/pipelines/radar/operators/include"
    "${PVA_SOLUTIONS_SRC_ROOT}/pipelines/radar/operators/priv"
)
