# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay (HCLTech).
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(online_video_reasoner LANGUAGES NONE)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_test(
    NAME online_video_reasoner_pytest
    COMMAND
      ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  set_tests_properties(
    online_video_reasoner_pytest
    PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}:$ENV{PYTHONPATH}"
      LABELS "operator;python;unit"
      TIMEOUT 120
  )
endif()
