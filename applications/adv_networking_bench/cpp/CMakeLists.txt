# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
  holoscan::ops::advanced_network_rx
  holoscan::ops::advanced_network_tx
  matx::matx
)

# Configure manager support through compile definitions
if(NOT DEFINED ANO_MGR)
    # Initialize ANO_MGR with a default if not provided
    # Note: rivermax not added to the default list yet since it requires licensing
    set(ANO_MGR "dpdk gpunetio" CACHE STRING "Manager type(s) list")
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

add_custom_target(adv_networking_bench_default_tx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_default_tx_yaml)

add_custom_target(adv_networking_bench_default_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_rx.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_default_rx_yaml)

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

# Installation
install(TARGETS adv_networking_bench
        DESTINATION examples/adv_networking_bench
        COMPONENT adv_networking_bench-cpp)

install(
  FILES ../adv_networking_bench_default_rx.yaml
        ../adv_networking_bench_default_tx.yaml
        ../adv_networking_bench_default_tx_rx.yaml
        ../adv_networking_bench_default_rx_multi_q.yaml
        ../adv_networking_bench_gpunetio_tx_rx.yaml
        ../adv_networking_bench_rmax_rx.yaml
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

  # Define a macro to handle the execution and checks
  macro(execute_and_check MSG CMD OUTPUT_VAR)
    execute_process(COMMAND bash "-c" "${CMD}"
                    RESULT_VARIABLE result_code
                    OUTPUT_VARIABLE ${OUTPUT_VAR}
                    ERROR_VARIABLE error_output
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT result_code EQUAL 0 OR "${${OUTPUT_VAR}}" STREQUAL "")
      message(STATUS "[ANO Test] ${MSG}")
      message(WARNING "'${CMD}' failed (${result_code}) or output was empty.\nError: ${error_output}")
      message(STATUS "[ANO Test] Skipping advanced networking test")
      return()
    else()
      message(STATUS "[ANO Test] ${MSG} ${${OUTPUT_VAR}}")
    endif()
  endmacro()

  # Check NIC addresses
  # Note: 0200 is the PCI-SIG class code for NICs. 15b3 is the Vendor ID for Mellanox.
  message(STATUS "[ANO Test] Checking system compatibility for advanced networking loopback test")
  execute_and_check("NVIDIA NIC BUS Addresses:" "lspci -n | awk '$2 == \"0200:\" && $3 ~ /^15b3:/ {print $1}'" cx_bus_address_list)
  string(REGEX REPLACE "\n" ";" cx_bus_address_list "${cx_bus_address_list}")
  list(LENGTH cx_bus_address_list cx_bus_address_list_count)
  if(NOT cx_bus_address_list_count EQUAL 2)
    message(WARNING "[ANO Test] Expected 2 CX NICs, found ${cx_bus_address_list_count}")
    message(WARNING "[ANO Test] Skipping advanced networking test")
    return()
  endif()

  # TEST 1 TX/RX loopback over single link. One TX queue and one RX queue
  list(GET cx_bus_address_list 0 tx_bus)
  list(GET cx_bus_address_list 1 rx_bus)
  execute_and_check("ETH Interface for ${tx_bus}:" "find /sys/class/net -exec bash -c 'echo {} $(readlink -f {}/device)' \; | grep ${tx_bus} | awk '{print $1}'" tx_if)
  execute_and_check("MAC Address for ${tx_if}:" "cat ${tx_if}/address" rx_eth_mac)
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml" TEST_CONFIG)
  string(REGEX REPLACE "# max_duration_ms:[^\n]*" "max_duration_ms: 20000" TEST_CONFIG ${TEST_CONFIG})
  string(REGEX REPLACE "address: [^\n]*interface[^\n]*Rx[^\n]*" "address: ${rx_bus}" TEST_CONFIG ${TEST_CONFIG})
  string(REGEX REPLACE "address: [^\n]*interface[^\n]*Tx[^\n]*" "address: ${tx_bus}" TEST_CONFIG ${TEST_CONFIG})
  string(REGEX REPLACE "address: [^\n]*Source[^\n]*Tx[^\n]*" "address: ${tx_bus}" TEST_CONFIG ${TEST_CONFIG})
  string(REGEX REPLACE "eth_dst_addr:[^\n]*" "eth_dst_addr: ${rx_eth_mac}" TEST_CONFIG ${TEST_CONFIG})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/adv_networking_bench_dpdk_multi_if_loopback_test.yaml" ${TEST_CONFIG})

  # TEST 2 RX multi queue with a single CPU core. Uses a scapy script to send packets
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_rx_multi_q.yaml" TEST_CONFIG)
  string(REGEX REPLACE "# max_duration_ms:[^\n]*" "max_duration_ms: 20000" TEST_CONFIG ${TEST_CONFIG})
  string(REGEX REPLACE "address: [^\n]*interface[^\n]*Rx[^\n]*" "address: ${rx_bus}" TEST_CONFIG ${TEST_CONFIG})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/adv_networking_bench_dpdk_rx_multi_q.yaml" ${TEST_CONFIG})


  # Define the tests
  file(COPY "testing/adv_networking_test_harness.py" DESTINATION "testing")
  add_test(NAME adv_networking_bench_dpdk_multi_if_loopback_test
           COMMAND python3 testing/adv_networking_test_harness.py "./adv_networking_bench adv_networking_bench_dpdk_multi_if_loopback_test.yaml"
                  --avg-throughput-threshold 90
                  --missed-packets-threshold 0.1
                  --port-map 0-1
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  add_test(NAME adv_networking_bench_dpdk_multi_rx_q_test
           COMMAND python3 testing/adv_networking_test_harness.py "./adv_networking_bench adv_networking_bench_dpdk_rx_multi_q.yaml"
                  --avg-throughput-threshold 0
                  --missed-packets-threshold 0
                  --packets-per-rx-queue 1 1
                  --port-map 0-0
                  --external-script ${CMAKE_CURRENT_SOURCE_DIR}/../utils/test_multi_rx_q.py
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
