# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(connext_dds NONE)

if(HOLOHUB_BUILD_PYTHON)
  set(connext_dds_python_files
    __init__.py
    publisher.py
    subscriber.py
  )

  list(TRANSFORM connext_dds_python_files
    PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/"
    OUTPUT_VARIABLE connext_dds_python_sources
  )

  if(DEFINED SKBUILD)
    # Standalone module wheel: install under the module namespace.
    set(connext_dds_python_output_dir "${CMAKE_BINARY_DIR}/holoscan/connext_dds")
  else()
    # HoloHub monorepo builds retain the standard holohub namespace.
    set(connext_dds_python_output_dir "${HOLOHUB_PYTHON_MODULE_OUT_DIR}/connext_dds")
  endif()
  file(MAKE_DIRECTORY "${connext_dds_python_output_dir}")

  list(TRANSFORM connext_dds_python_files
    PREPEND "${connext_dds_python_output_dir}/"
    OUTPUT_VARIABLE connext_dds_python_outputs
  )

  add_custom_target(connext_dds_python ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${connext_dds_python_sources}
      "${connext_dds_python_output_dir}"
    DEPENDS ${connext_dds_python_sources}
    BYPRODUCTS ${connext_dds_python_outputs}
  )

  if(DEFINED SKBUILD)
    install(
      DIRECTORY "${CMAKE_BINARY_DIR}/holoscan"
      DESTINATION "."
      PATTERN "__pycache__" EXCLUDE
    )
  endif()

  if(BUILD_TESTING)
    include(add_python_tests)
    add_python_tests(
      INPUT tests
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
      PREFIX connext_dds
    )
  endif()
endif()
