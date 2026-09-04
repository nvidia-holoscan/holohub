# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(dds_pubsub NONE)

if(HOLOHUB_BUILD_PYTHON)
  set(dds_pubsub_python_files
    python/dds/__init__.py
    python/dds/publisher.py
    python/dds/subscriber.py
  )

  list(TRANSFORM dds_pubsub_python_files
    PREPEND "${CMAKE_CURRENT_SOURCE_DIR}/"
    OUTPUT_VARIABLE dds_pubsub_python_sources
  )

  set(dds_pubsub_python_output_dir "${HOLOHUB_PYTHON_MODULE_OUT_DIR}/dds")
  file(MAKE_DIRECTORY "${dds_pubsub_python_output_dir}")

  set(dds_pubsub_python_output_files
    __init__.py
    publisher.py
    subscriber.py
  )

  list(TRANSFORM dds_pubsub_python_output_files
    PREPEND "${dds_pubsub_python_output_dir}/"
    OUTPUT_VARIABLE dds_pubsub_python_outputs
  )

  add_custom_target(dds_pubsub_python ALL
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${dds_pubsub_python_sources}
      "${dds_pubsub_python_output_dir}"
    DEPENDS ${dds_pubsub_python_sources}
    BYPRODUCTS ${dds_pubsub_python_outputs}
  )

  if(BUILD_TESTING)
    include(add_python_tests)
    add_python_tests(
      INPUT tests
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python"
      PREFIX dds_pubsub
    )
  endif()
endif()
