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

# The PVA-SDK supports a development environment on x86_64 hosts where the PVA device code is run on
# an emulator (default PVA_BUILD_MODE=NATIVE). This is NOT performant, but is useful for debugging and
# development.
# The PVA-SDK also supports cross-compiling code for Jetson devices using PVA_BUILD_MODE=L4T on an x86_64 host.
# However, the holoscan development environment does not support cross-compile.  The holoscan approach is to
# build directly for the target, on the target.

# For the purpose of this example application, when the system processor is aarch64 we assume it is a Jetson device, and
# instead of finding the full PVA-SDK, we only search for the required runtime libraries and header files needed to
# build the operators
# See the Dockerfile for more details about where the required dependencies are picked up from the target environment.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
  # on Jetson target device we define the cupva_host target by finding the required runtime lib and header files manually
  if(EXISTS /opt/nvidia/pva-sdk-2.8/lib/aarch64-linux-gnu/libcupva_host.so.2.8)
    add_library(cupva_host SHARED IMPORTED)
    set_target_properties(cupva_host PROPERTIES
      IMPORTED_LOCATION /opt/nvidia/pva-sdk-2.8/lib/aarch64-linux-gnu/libcupva_host.so.2.8
      INTERFACE_INCLUDE_DIRECTORIES /opt/nvidia/pva-sdk-2.8/include
    )
    set(pva-sdk_FOUND TRUE)
  else()
    set(pva-sdk_FOUND FALSE)
  endif()
else()
  # the pva-sdk package provides cupva_host target for the cupva host runtime
  find_package(pva-sdk 2.8.0)
endif()

if(NOT pva-sdk_FOUND)
  message(WARNING "Cannot find pva-sdk dependencies for pva_radar operators. Skipping configuration.")
  return()
endif()

# Set up pva-solutions dependencies. Default path is where the Dockerfile installed the source tree.
# It can be overridden with a different path by mounting the source tree to the container and passing the
# new path as a cmake argument at build time.
set(PVA_SOLUTIONS_SRC_ROOT "/workspace/pva-solutions" CACHE PATH "Path to pva-solutions source tree")
include(pva-solutions-config.cmake)

if(NOT pva-solutions_FOUND)
  message(WARNING "Cannot find pva-solutions dependencies for pva_radar operators. Skipping configuration.")
  return()
endif()

find_package(holoscan 2.6.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_holohub_operator(pva_radar)
add_holohub_operator(pva_radar_graphics)
add_holohub_operator(raw_radar_cube_source)
