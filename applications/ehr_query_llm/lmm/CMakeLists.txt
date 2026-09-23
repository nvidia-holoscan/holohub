# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(lmm LANGUAGES NONE)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME ehr_query_llm_webserver_test
           COMMAND "${Python3_EXECUTABLE}" -m pytest -q
                   "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_ehr_webserver.py"
           WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  set_tests_properties(ehr_query_llm_webserver_test PROPERTIES
                       LABELS "application;python;unit"
                       TIMEOUT 60)
endif()
