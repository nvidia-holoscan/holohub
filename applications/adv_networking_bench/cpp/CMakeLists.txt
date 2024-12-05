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
  ## adv_networking_bench_default_tx_rx_test ##
  # The test below is expected to run on an IGX devkit with a loopback
  # connecting both ports of the IGX NIC (default config currently hardcodes
  # the BUS address of these two ports), with optimal system setups (BAR1,
  # GPU clock, CPU clock, huge pages, etc... docs incoming).

  # Get the BUS ID of the interface doing Rx in the config
  execute_process(COMMAND bash "-c" "grep -A2 -B2 ' rx:' ${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml | grep ' address:' | awk '{print $2}'"
                  OUTPUT_VARIABLE rx_bus
                  COMMAND_ERROR_IS_FATAL ANY
                  OUTPUT_STRIP_TRAILING_WHITESPACE)
  message(STATUS "[ANO Test] Rx interface BUS info from config: ${rx_bus}")

  if(rx_bus)
    # Get the interface name
    execute_process(COMMAND bash "-c" "find /sys/class/net -exec bash -c 'echo {} $(readlink -f {}/device)' \; | grep ${rx_bus} | awk '{print $1}'"
                    OUTPUT_VARIABLE rx_if
                    COMMAND_ERROR_IS_FATAL ANY
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    message(STATUS "[ANO Test] Interface for ${rx_bus}: ${rx_if}")

    if(rx_if)
      # Get the ETH mac address
      execute_process(COMMAND cat "${rx_if}/address"
                      OUTPUT_VARIABLE rx_eth_mac
                      COMMAND_ERROR_IS_FATAL ANY
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
      message(STATUS "[ANO Test] MAC address for interface ${rx_if}: ${rx_eth_mac}")

      # Inject in test config + add app timeout
      file(READ "${CMAKE_CURRENT_SOURCE_DIR}/../adv_networking_bench_default_tx_rx.yaml" TEST_CONFIG)
      string(REGEX REPLACE "eth_dst_addr:[^\n]*" "eth_dst_addr: ${rx_eth_mac}" TEST_CONFIG ${TEST_CONFIG})
      string(REGEX REPLACE "# max_duration_ms:[^\n]*" "max_duration_ms: 20000" TEST_CONFIG ${TEST_CONFIG})
      file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/adv_networking_bench_default_tx_rx_test.yaml" ${TEST_CONFIG})

      # Create test
      file(COPY "testing/adv_networking_test_harness.py" DESTINATION "testing")
      add_test(NAME adv_networking_bench_dpdk_test
              COMMAND python3 testing/adv_networking_test_harness.py "./adv_networking_bench adv_networking_bench_default_tx_rx_test.yaml" --avg-throughput-threshold 90 --missed-packets-threshold 0.1
              WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
    else()
      message(STATUS "Rx Interface is empty - skipping test")
    endif()
  else()
    message(STATUS "Rx Bud Id is empty - skipping test")
    endif
  endif()
endif()
