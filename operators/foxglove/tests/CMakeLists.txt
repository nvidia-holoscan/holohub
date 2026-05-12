# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(GTest QUIET)
if(NOT GTest_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
    GIT_SHALLOW TRUE
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

add_executable(test_foxglove
  test_foxglove_publisher.cpp
)

target_link_libraries(test_foxglove
  PRIVATE
    foxglove
    GTest::gtest
    GTest::gtest_main
)

add_test(NAME foxglove_unit_tests COMMAND test_foxglove)
set_tests_properties(foxglove_unit_tests PROPERTIES
  LABELS "unit;foxglove"
  TIMEOUT 30
)
