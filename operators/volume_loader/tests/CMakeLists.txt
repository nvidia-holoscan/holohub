# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/releases/download/v1.17.0/googletest-1.17.0.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
  )
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(volume_loader_mhd_test
  test_mhd_loader.cpp
)

target_link_libraries(volume_loader_mhd_test
  PRIVATE
    holoscan::core
    holoscan::ops::volume_loader
    GTest::gtest_main
)

add_test(NAME volume_loader_mhd_test COMMAND volume_loader_mhd_test)
set_tests_properties(volume_loader_mhd_test PROPERTIES
  LABELS "unit;volume_loader"
  TIMEOUT 30
)
