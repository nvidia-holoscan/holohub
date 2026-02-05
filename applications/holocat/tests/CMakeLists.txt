# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION &
# AFFILIATES. All rights reserved.
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

# C++ Unit Tests for HoloCat Application

# Find required packages
find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Use FetchContent to get GTest if not found
find_package(GTest QUIET)

if(NOT GTest_FOUND)
  message(STATUS "GTest not found, fetching from GitHub...")
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE
  )
  # For Windows: Prevent overriding the parent project's compiler/linker
  # settings (googletest expects lowercase name)
  # cmake-lint: disable=C0103
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)

  # Create aliases to match find_package targets
  if(NOT TARGET GTest::gtest)
    add_library(GTest::gtest ALIAS gtest)
  endif()
  if(NOT TARGET GTest::gtest_main)
    add_library(GTest::gtest_main ALIAS gtest_main)
  endif()
endif()

# Common include directories for all tests
set(TEST_INCLUDE_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/..
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mocks
)

# Create a library with the operator implementations (without main)
add_library(holocat_operators
  ../cpp/hc_data_tx_op.cpp
  ../cpp/hc_data_rx_op.cpp
  ../cpp/hc_data_tx_op.hpp
  ../cpp/hc_data_rx_op.hpp
  ../cpp/holocat_config.hpp
)

target_include_directories(holocat_operators PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/mocks
)

target_link_libraries(holocat_operators PUBLIC
  holoscan::core
)

# =============================================================================
# Unit Tests - TX/RX Operators (no EC-Master dependency)
# =============================================================================

add_executable(test_hc_data_tx_op
  test_hc_data_tx_op.cpp
)

target_link_libraries(test_hc_data_tx_op
  PRIVATE
    holocat_operators
    holoscan::core
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(test_hc_data_tx_op PRIVATE ${TEST_INCLUDE_DIRS})

add_test(NAME holocat_unit.hc_data_tx_op
         COMMAND test_hc_data_tx_op)

set_tests_properties(holocat_unit.hc_data_tx_op PROPERTIES
  TIMEOUT 30
  LABELS "unit;holocat"
)

# -----------------------------------------------------------------------------

add_executable(test_hc_data_rx_op
  test_hc_data_rx_op.cpp
)

target_link_libraries(test_hc_data_rx_op
  PRIVATE
    holocat_operators
    holoscan::core
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(test_hc_data_rx_op PRIVATE ${TEST_INCLUDE_DIRS})

add_test(NAME holocat_unit.hc_data_rx_op
         COMMAND test_hc_data_rx_op)

set_tests_properties(holocat_unit.hc_data_rx_op PROPERTIES
  TIMEOUT 30
  LABELS "unit;holocat"
)

# =============================================================================
# Mocked Integration Tests (no EC-Master SDK required)
# =============================================================================

add_executable(test_integration_mocked
  test_integration_mocked.cpp
)

target_link_libraries(test_integration_mocked
  PRIVATE
    holocat_operators
    holoscan::core
    GTest::gtest
    GTest::gtest_main
)

target_include_directories(test_integration_mocked PRIVATE ${TEST_INCLUDE_DIRS})

add_test(NAME holocat_mocked.integration
         COMMAND test_integration_mocked)

set_tests_properties(holocat_mocked.integration PROPERTIES
  TIMEOUT 60
  LABELS "integration;holocat;mocked"
)

# =============================================================================
# HolocatOp Tests (requires EC-Master SDK or mocks)
# =============================================================================

# Only build if EC-Master SDK is available
if(EcMaster_FOUND)
  # Create a library with HolocatOp that can be tested
  add_library(holocat_op_lib
    ../cpp/holocat_op.cpp
    ../cpp/holocat_op.hpp
  )

  target_include_directories(holocat_op_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/../cpp
    ${ECMASTER_INCLUDE_DIRS}
  )

  target_link_libraries(holocat_op_lib PUBLIC
    holoscan::core
    ${ECMASTER_LIBRARIES}
    pthread
    rt
  )

  target_compile_options(holocat_op_lib PRIVATE
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-packed-not-aligned
  )

  # HolocatOp basic tests (configuration validation)
  add_executable(test_holocat_op
    test_holocat_op.cpp
  )

  target_link_libraries(test_holocat_op
    PRIVATE
      holocat_op_lib
      holocat_operators
      holoscan::core
      GTest::gtest
      GTest::gtest_main
  )

  target_include_directories(test_holocat_op PRIVATE
    ${TEST_INCLUDE_DIRS}
    ${ECMASTER_INCLUDE_DIRS}
  )

  add_test(NAME holocat_unit.holocat_op
           COMMAND test_holocat_op)

  set_tests_properties(holocat_unit.holocat_op PROPERTIES
    TIMEOUT 30
    LABELS "unit;holocat"
  )

  # ===========================================================================
  # Hardware Integration Tests (optional, requires hardware)
  # ===========================================================================

  add_executable(test_integration_hardware
    test_integration_hardware.cpp
  )

  target_link_libraries(test_integration_hardware
    PRIVATE
      holocat_op_lib
      holocat_operators
      holoscan::core
      GTest::gtest
      GTest::gtest_main
  )

  target_include_directories(test_integration_hardware PRIVATE
    ${TEST_INCLUDE_DIRS}
    ${ECMASTER_INCLUDE_DIRS}
  )

  add_test(NAME holocat_hardware.integration
           COMMAND test_integration_hardware)

  set_tests_properties(holocat_hardware.integration PROPERTIES
    TIMEOUT 120
    LABELS "hardware;holocat;integration"
  )

else()
  message(
    STATUS
    "EC-Master SDK not found - skipping HolocatOp and hardware tests")
endif()

# Set output directory for all test executables
set_target_properties(
  test_hc_data_tx_op
  test_hc_data_rx_op
  test_integration_mocked
  PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY
    "${CMAKE_BINARY_DIR}/applications/holocat/tests"
)

if(EcMaster_FOUND)
  set_target_properties(
    test_holocat_op
    test_integration_hardware
    PROPERTIES
      RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/applications/holocat/tests"
  )
endif()
