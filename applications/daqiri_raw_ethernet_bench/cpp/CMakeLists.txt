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

cmake_minimum_required(VERSION 3.24)
project(daqiri_raw_ethernet_bench CXX CUDA)

# Dependencies
find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
if(DEFINED ENV{DAQIRI_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{DAQIRI_ROOT}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "/opt/daqiri")
find_package(daqiri REQUIRED CONFIG)
include(HoloHubDaqiri)

# Create the executable
add_executable(daqiri_raw_ethernet_bench
  kernels.cu
  main.cpp
)
target_link_libraries(daqiri_raw_ethernet_bench
  PRIVATE
  holoscan::core
)
holohub_link_daqiri(daqiri_raw_ethernet_bench)

# Configure engine support through compile definitions
if(NOT DEFINED DAQIRI_ENGINE)
    set(DAQIRI_ENGINE "dpdk" CACHE STRING "DAQIRI engine type(s) list")
endif()
separate_arguments(DAQIRI_ENGINE_LIST UNIX_COMMAND ${DAQIRI_ENGINE})
foreach(ENGINE IN LISTS DAQIRI_ENGINE_LIST)
    string(TOUPPER "${ENGINE}" ENGINE_UPPER)
    message(STATUS "Adding DAQIRI engine ${ENGINE} to the application configuration")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "DAQIRI_ENGINE_${ENGINE_UPPER}=1")
endforeach()

# Copy config file
set(DAQIRI_RAW_ETHERNET_BENCH_CONFIGS
  daqiri_raw_ethernet_bench_default_tx_rx.yaml
)

foreach(CONFIG_FILE IN LISTS DAQIRI_RAW_ETHERNET_BENCH_CONFIGS)
  string(REPLACE "." "_" CONFIG_TARGET "${CONFIG_FILE}")
  add_custom_target(${CONFIG_TARGET}
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../${CONFIG_FILE}" ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../${CONFIG_FILE}"
  )
  add_dependencies(daqiri_raw_ethernet_bench ${CONFIG_TARGET})
endforeach()

# Installation
install(TARGETS daqiri_raw_ethernet_bench
        DESTINATION examples/daqiri_raw_ethernet_bench
        COMPONENT daqiri_raw_ethernet_bench-cpp)

install(
  FILES
    ../daqiri_raw_ethernet_bench_default_tx_rx.yaml
  DESTINATION examples/daqiri_raw_ethernet_bench
  COMPONENT daqiri_raw_ethernet_bench-configs
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)

install(
  FILES CMakeLists.txt.install
  RENAME CMakeLists.txt
  DESTINATION examples/daqiri_raw_ethernet_bench
  COMPONENT daqiri_raw_ethernet_bench-cppsrc
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)

install(
  FILES
    main.cpp
    kernels.cu
    kernels.cuh
    default_bench_op_rx.h
    default_bench_op_tx.h
  DESTINATION examples/daqiri_raw_ethernet_bench
  COMPONENT daqiri_raw_ethernet_bench-cppsrc
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)


# Add testing
if(BUILD_TESTING)
  # Copy the test files to the build directory
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/benchmark_utils.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/io_utils.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/process_utils.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/test_daqiri_raw_ethernet_bench.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/conftest.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/nvidia_nic_utils.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/yaml_config_utils.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")

  # Register individual pytest tests with CTest
  include(add_python_tests)
  add_python_tests(
    INPUT "testing"
    ENVIRONMENT "DAQIRI_ENGINE_LIST=${DAQIRI_ENGINE_LIST}"
    PYTEST_ARGS "-v" "-s" "--log-cli-level=DEBUG" "--show-capture=no"
  )
endif()
