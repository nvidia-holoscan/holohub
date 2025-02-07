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

cmake_minimum_required(VERSION 3.20)
project(adv_networking_bench CXX CUDA)

# Dependencies
find_package(holoscan 2.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG v0.8.0
)
FetchContent_MakeAvailable(MatX)

# Global variables
set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

# Create the executable
add_executable(adv_networking_bench
  kernels.cu
  main.cpp
)
target_link_libraries(adv_networking_bench
  PRIVATE
  holoscan::core
  holoscan::advanced_network
  matx::matx
)

# Configure manager support through compile definitions
if(NOT DEFINED ANO_MGR)
    # Initialize ANO_MGR with a default if not provided
    # Note: rivermax not added to the default list yet since it requires licensing
    set(ANO_MGR "dpdk gpunetio rdma" CACHE STRING "Manager type(s) list")
endif()
separate_arguments(ANO_MGR_LIST UNIX_COMMAND ${ANO_MGR})
foreach(MGR IN LISTS ANO_MGR_LIST)
    message(STATUS "Adding manager ${MGR} to the configuration")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "ANO_MGR_${MGR}=1")
endforeach()

# Copy config file
add_custom_target(adv_networking_bench_default_tx_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_default_tx_rx_yaml)

add_custom_target(adv_networking_bench_default_rx_multi_q_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_rx_multi_q.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_rx_multi_q.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_default_rx_multi_q_yaml)

add_custom_target(adv_networking_bench_default_tx_rx_hds_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx_hds.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx_hds.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_default_tx_rx_hds_yaml)

add_custom_target(adv_networking_bench_gpunetio_tx_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_gpunetio_tx_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_gpunetio_tx_rx.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_gpunetio_tx_rx_yaml)

add_custom_target(adv_networking_bench_rmax_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_rmax_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_rmax_rx.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_rmax_rx_yaml)

add_custom_target(adv_networking_bench_rdma_tx_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_rdma_tx_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_rdma_tx_rx.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_rdma_tx_rx_yaml)

# Installation
install(TARGETS adv_networking_bench
        DESTINATION examples/adv_networking_bench
        COMPONENT adv_networking_bench-cpp)

install(
  FILES
    ../adv_networking_bench_default_tx_rx_hds.yaml
    ../adv_networking_bench_default_tx_rx.yaml
    ../adv_networking_bench_default_rx_multi_q.yaml
    ../adv_networking_bench_gpunetio_tx_rx.yaml
    ../adv_networking_bench_rmax_rx.yaml
    ../adv_networking_bench_rdma_tx_rx.yaml
  DESTINATION examples/adv_networking_bench
  COMPONENT adv_networking_bench-configs
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)

install(
  FILES CMakeLists.txt.install
  RENAME CMakeLists.txt
  DESTINATION examples/adv_networking_bench
  COMPONENT adv_networking_bench-cppsrc
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
    doca_bench_op_rx.h
    doca_bench_op_tx.h
  DESTINATION examples/adv_networking_bench
  COMPONENT adv_networking_bench-cppsrc
  PERMISSIONS OWNER_READ OWNER_WRITE
              GROUP_READ GROUP_WRITE
              WORLD_READ WORLD_WRITE
)


# Add testing
if(BUILD_TESTING)
  # Copy the test files to the build directory
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/benchmark_utils.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/process_utils.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/test_ano_bench.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/conftest.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/nvidia_nic_utils.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")
  file(COPY "${CMAKE_CURRENT_SOURCE_DIR}/testing/yaml_config_utils.py" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/testing")

  # Register individual pytest tests with CTest
  include(add_python_tests)
  add_python_tests(
    INPUT "testing"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    PYTEST_ARGS "-v" "-s" "--log-cli-level=DEBUG" "--show-capture=no"
  )
endif()
