# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay (HCLTech).
# SPDX-License-Identifier: Apache-2.0

project(procedural_narrator NONE)

find_package(
  holoscan 4.4 REQUIRED CONFIG
  PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install"
)

if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL nvidia/clara-holoscan/holoscan_endoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 480
    GXF_ENTITIES_WIDTH 854
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )
endif()

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(
    NAME procedural_narrator_cli
    COMMAND
      ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/procedural_narrator.py
      --help
  )
  add_test(
    NAME procedural_narrator_pytest
    COMMAND
      ${Python3_EXECUTABLE} -m pytest -q
      ${CMAKE_CURRENT_SOURCE_DIR}
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
  set_tests_properties(
    procedural_narrator_cli
    procedural_narrator_pytest
    PROPERTIES
      ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}:$ENV{PYTHONPATH}"
  )
  set_tests_properties(
    procedural_narrator_cli
    PROPERTIES
      PASS_REGULAR_EXPRESSION "Narrate a clinical replay or camera feed"
      TIMEOUT 60
  )
  set_tests_properties(
    procedural_narrator_pytest
    PROPERTIES
      LABELS "application;python;unit"
      TIMEOUT 60
  )
  if(HOLOHUB_DOWNLOAD_DATASETS)
    add_test(
      NAME procedural_narrator_functional
      COMMAND
        ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/procedural_narrator_functional.py
        --data ${HOLOHUB_DATA_DIR}/endoscopy
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    )
    set_tests_properties(
      procedural_narrator_functional
      PROPERTIES
        ENVIRONMENT "PYTHONPATH=${CMAKE_SOURCE_DIR}:$ENV{PYTHONPATH}"
        LABELS "application;functional;gpu"
        PASS_REGULAR_EXPRESSION "SUCCESS: replay, temporal reasoner"
        TIMEOUT 90
    )
  endif()
endif()
