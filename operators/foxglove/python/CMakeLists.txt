# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

include(pybind11_add_holohub_module)
pybind11_add_holohub_module(
  CPP_CMAKE_TARGET foxglove
  CLASS_NAME "FoxglovePublisherOp"
  SOURCES foxglove_publisher.cpp
)

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py
  ${HOLOHUB_PYTHON_MODULE_OUT_DIR}/foxglove/__init__.py
  COPYONLY
)

if(BUILD_TESTING)
  if(NOT DEFINED GXF_LIB_DIR)
    get_filename_component(GXF_LIB_DIR "${holoscan_DIR}/../.." ABSOLUTE)
  endif()

  set(PYTHON_TEST_ENV
      "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
  )

  add_test(
    NAME foxglove_python_import_test
    COMMAND ${Python3_EXECUTABLE} -c
            "from holohub.foxglove import FoxgloveCompressedVideo, FoxgloveFrameTransform, FoxglovePoseAdapterOp, FoxglovePublisherOp, FoxgloveTensorAdapterOp, PackedElementField; field = PackedElementField(); field.name = 'x'; field.offset = 0; field.type = PackedElementField.NumericType.FLOAT32; video = FoxgloveCompressedVideo(); video.data = b'\\x00\\x00\\x01'; transform = FoxgloveFrameTransform(); assert FoxglovePublisherOp.__name__ == 'FoxglovePublisherOp'; assert FoxglovePoseAdapterOp.__name__ == 'FoxglovePoseAdapterOp'; assert field.type == PackedElementField.NumericType.FLOAT32; assert video.data == b'\\x00\\x00\\x01'; assert transform.rotation[3] == 1.0"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(foxglove_python_import_test PROPERTIES
    ENVIRONMENT "${PYTHON_TEST_ENV}"
    LABELS "unit;python;foxglove"
  )
endif()
