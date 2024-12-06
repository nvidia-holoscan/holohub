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

find_package(holoscan 0.6 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_package(PkgConfig REQUIRED)

include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG v0.8.0
)
FetchContent_MakeAvailable(MatX)

add_executable(adv_networking_bench
  kernels.cu
  main.cpp
)

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")

target_link_libraries(adv_networking_bench
  PRIVATE
  holoscan::core
  holoscan::advanced_network_rx
  holoscan::advanced_network_tx
  matx::matx
)
# Initialize ANO_MGR with a default if not provided
if(NOT DEFINED ANO_MGR)
    set(ANO_MGR "dpdk doca" CACHE STRING "Manager type(s) list")
endif()

separate_arguments(ANO_MGR_LIST UNIX_COMMAND ${ANO_MGR})

# Define targets for each manager type
foreach(MGR IN LISTS ANO_MGR_LIST)
    message(STATUS "Adding manager ${MGR} to the configuration")
    target_compile_definitions(${PROJECT_NAME} PUBLIC "ANO_MGR_${MGR}=1")
endforeach()

set_target_properties(adv_networking_bench PROPERTIES CUDA_ARCHITECTURES "80;90")

# Copy config file
add_custom_target(adv_networking_bench_default_tx_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_default_tx_rx_yaml)

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

add_custom_target(adv_networking_bench_doca_tx_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_doca_tx_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_doca_tx_rx.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_doca_tx_rx_yaml)

add_custom_target(adv_networking_bench_rmax_rx_yaml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_rmax_rx.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_rmax_rx.yaml"
)
add_dependencies(adv_networking_bench adv_networking_bench_rmax_rx_yaml)

# Installation
install(TARGETS adv_networking_bench
        DESTINATION bin/adv_networking_bench/cpp)

install(
  FILES ../adv_networking_bench_default_rx.yaml
        ../adv_networking_bench_default_tx.yaml
        ../adv_networking_bench_default_tx_rx.yaml
        ../adv_networking_bench_doca_tx_rx.yaml
        ../adv_networking_bench_rmax_rx_yaml
  DESTINATION bin/adv_networking_bench/cpp
)

# Add testing
if(BUILD_TESTING)
  ## adv_networking_bench_dpdk_loopback_test ##
  # The test below is expected to run on a system with a ConnectX NIC with
  # both ports connected in a loopback, with optimal system setups (BAR1,
  # GPU clock, CPU clock, huge pages, etc... docs incoming).

  # Define a macro to handle the execution and checks
  macro(execute_and_check MSG CMD OUTPUT_VAR)
    execute_process(COMMAND bash "-c" "${CMD}"
                    RESULT_VARIABLE result_code
                    OUTPUT_VARIABLE ${OUTPUT_VAR}
                    ERROR_VARIABLE error_code
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT result_code EQUAL 0 OR "${${OUTPUT_VAR}}" STREQUAL "")
      message(WARNING "[ANO Test] ${MSG} '${CMD}' failed (${result_code}) or output was empty")
      message(STATUS "[ANO Test] Skipping advanced networking loopback test")
      return()
    else()
      message(STATUS "[ANO Test] ${MSG} ${${OUTPUT_VAR}}")
    endif()
  endmacro()

  # Check NIC addresses
  message(STATUS "[ANO Test] Checking system compatibility for advanced networking loopback test")
  execute_and_check("ConnectX BUS Addresses:" "lspci | grep -E \"Ethernet.*ConnectX\" | awk '{print $1}'" cx_bus_address_list)
  string(REGEX REPLACE "\n" ";" cx_bus_address_list "${cx_bus_address_list}")
  list(LENGTH cx_bus_address_list cx_bus_address_list_count)
  if(NOT cx_bus_address_list_count EQUAL 2)
    message(WARNING "[ANO Test] Expected 2 CX NICs, found ${cx_bus_address_list_count}")
    message(WARNING "[ANO Test] Skipping advanced networking test")
    return()
  endif()
  list(GET cx_bus_address_list 0 tx_bus)
  list(GET cx_bus_address_list 1 rx_bus)
  execute_and_check("ETH Interface for ${tx_bus}:" "find /sys/class/net -exec bash -c 'echo {} $(readlink -f {}/device)' \; | grep ${tx_bus} | awk '{print $1}'" tx_if)
  execute_and_check("MAC Address for ${tx_if}:" "cat ${tx_if}/address" rx_eth_mac)

  # Inject in the test config and add app timeout
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml" TEST_CONFIG)
  string(REGEX REPLACE "# max_duration_ms:[^\n]*" "max_duration_ms: 20000" TEST_CONFIG ${TEST_CONFIG})
  string(REGEX REPLACE "address: [^\n]*interface[^\n]*Rx[^\n]*" "address: ${rx_bus}" TEST_CONFIG ${TEST_CONFIG})
  string(REGEX REPLACE "address: [^\n]*interface[^\n]*Tx[^\n]*" "address: ${tx_bus}" TEST_CONFIG ${TEST_CONFIG})
  string(REGEX REPLACE "address: [^\n]*Source[^\n]*Tx[^\n]*" "address: ${tx_bus}" TEST_CONFIG ${TEST_CONFIG})
  string(REGEX REPLACE "eth_dst_addr:[^\n]*" "eth_dst_addr: ${rx_eth_mac}" TEST_CONFIG ${TEST_CONFIG})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/adv_networking_bench_dpdk_loopback_test.yaml" ${TEST_CONFIG})

  # Define the test
  file(COPY "testing/adv_networking_test_harness.py" DESTINATION "testing")
  add_test(NAME adv_networking_bench_dpdk_test
           COMMAND python3 testing/adv_networking_test_harness.py "./adv_networking_bench adv_networking_bench_dpdk_loopback_test.yaml" --avg-throughput-threshold 90 --missed-packets-threshold 0.1
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
endif()
