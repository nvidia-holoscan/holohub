# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.20.1)
# Check if we can find a CXX compiler before enabling it
find_program(CXX_COMPILER_FOUND NAMES g++ clang++ c++)
if(CXX_COMPILER_FOUND)
  project(Holohub CXX)
else()
  project(Holohub NONE)
endif()

# set CMAKE_MODULE_PATH
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/modules)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/deps)

# To configure which apps/operators to build
include(HoloHubConfigHelpers)

# Set install directory
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/install" CACHE PATH "Installation directory" FORCE)
endif()

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
endif()

# Enable Testing
include(CTest)

# Options
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
option(HOLOHUB_BUILD_CPP "Build support for C++" ON)
option(HOLOHUB_BUILD_PYTHON "Build support for Python" ON)

# Enable flow benchmarking
option(FLOW_BENCHMARKING "Enable Flow Benchmarking" OFF)
if(FLOW_BENCHMARKING)
  set(FLOW_BENCHMARK_DIR "${CMAKE_BINARY_DIR}/flow_benchmarks")
  file(MAKE_DIRECTORY "${FLOW_BENCHMARK_DIR}")
endif()

if(HOLOHUB_BUILD_PYTHON)
  # Define and create a directory to store python operators under the holohub python module
  if(NOT CMAKE_INSTALL_LIBDIR)
    set(CMAKE_INSTALL_LIBDIR lib)
  endif()
  set(HOLOHUB_PYTHON_MODULE_OUT_DIR ${CMAKE_BINARY_DIR}/python/${CMAKE_INSTALL_LIBDIR}/holohub)
  file(MAKE_DIRECTORY ${HOLOHUB_PYTHON_MODULE_OUT_DIR})

  # Find the Python interpreter
  find_package(Python COMPONENTS Interpreter)
  set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
  # Set the Python executable for testing if not defined
  if(NOT DEFINED PYTHON_EXECUTABLE_TEST)
    set(PYTHON_EXECUTABLE_TEST ${Python_EXECUTABLE})
  endif()
endif()

# Set the default data directory
set(HOLOHUB_DATA_DIR "${CMAKE_BINARY_DIR}/data" CACHE PATH "Data Download directory")
set(CMAKE_INSTALL_RPATH
    "\$ORIGIN:\$ORIGIN/../../../lib:\$ORIGIN/../../lib:/opt/nvidia/holoscan/lib/")

# Set the default CUDA target platforms
if(NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES "native")
endif()

# External Holoscan Module manifest.
#
# The HoloHub CLI writes ${CMAKE_BINARY_DIR}/external_operators_manifest.cmake
# before configure runs. The manifest is data-only — FetchContent_Declare
# entries (lazy; nothing fetches yet) and HOLOHUB_EXT_OP_<op>_PROVIDER
# variables that map operator names to providing modules. The end-of-file
# block below scans those after the rest of the tree has registered which
# OP_<op>=ON, and FetchContent_MakeAvailable's only the modules whose
# operators ended up enabled.
include(${CMAKE_BINARY_DIR}/external_operators_manifest.cmake OPTIONAL)

# Build packaging
add_subdirectory(pkg)

# Build the workflows
add_subdirectory(workflows)

# Build the applications
add_subdirectory(applications)

# Build the benchmarks
add_subdirectory(benchmarks)

# Build the operators
add_subdirectory(operators)

# Build the extensions
add_subdirectory(gxf_extensions)

# ── Resolve external operators that ended up enabled ──────────────────
#
# Walk normal variables for HOLOHUB_EXT_OP_<op>_PROVIDER entries (set by the
# manifest above). For each operator whose OP_<x> option is ON,
# collect its providing module. Then FetchContent_MakeAvailable each unique
# module exactly once — that fetches sources lazily and add_subdirectory's
# the module's top-level CMakeLists. The module's per-operator gating
# (PROJECT_IS_TOP_LEVEL=FALSE → BUILD_ALL=OFF, plus the cache-FORCE'd
# OP_<op>=ON values from add_holohub_application(... DEPENDS OPERATORS …))
# means only the requested operators in the fetched module actually build.
get_cmake_property(_holohub_all_vars VARIABLES)
set(_holohub_ext_modules_needed "")
foreach(_var IN LISTS _holohub_all_vars)
  if(_var MATCHES "^HOLOHUB_EXT_OP_(.+)_PROVIDER$")
    set(_op_name "${CMAKE_MATCH_1}")
    if(OP_${_op_name})
      list(APPEND _holohub_ext_modules_needed "${${_var}}")
    endif()
  endif()
endforeach()

if(_holohub_ext_modules_needed)
  list(REMOVE_DUPLICATES _holohub_ext_modules_needed)
  include(FetchContent)
  foreach(_provider IN LISTS _holohub_ext_modules_needed)
    message(STATUS "Fetching external Holoscan Module: ${_provider}")
    FetchContent_MakeAvailable(${_provider})
  endforeach()
endif()

option(BUILD_HOLOHUB_TESTING OFF)
if(BUILD_TESTING AND BUILD_HOLOHUB_TESTING)
  enable_testing()
  # The HoloHub CLI implementation is now the standalone ``holoscan-cli``
  # package; its unit tests live in that repository. The only HoloHub-side
  # CLI test still in this tree is the consolidation smoke test, which is
  # invoked directly via ``pytest utilities/cli/tests/test_holoscan_cli_consolidation.py``
  # against the installed holoscan-cli.
endif()
