# SPDX-FileCopyrightText: Copyright (c) 2026 Real-Time Innovations, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

function(add_connext_dds_example APPLICATION_NAME APPLICATION_SOURCE)
  set(COMMON_DIR "${CMAKE_CURRENT_FUNCTION_LIST_DIR}")

  # Connext automatically loads this file from the application's working directory.
  configure_file("${COMMON_DIR}/USER_QOS_PROFILES.xml" USER_QOS_PROFILES.xml COPYONLY)
  configure_file("${COMMON_DIR}/connext_dds_common.py" connext_dds_common.py COPYONLY)
  configure_file("${CMAKE_CURRENT_SOURCE_DIR}/${APPLICATION_SOURCE}" ${APPLICATION_SOURCE} COPYONLY)

  add_custom_target(${APPLICATION_NAME} ALL
    DEPENDS dds_pubsub_python
  )

  install(
    FILES
      "${CMAKE_CURRENT_SOURCE_DIR}/${APPLICATION_SOURCE}"
      "${COMMON_DIR}/connext_dds_common.py"
      "${COMMON_DIR}/USER_QOS_PROFILES.xml"
    DESTINATION "examples/${APPLICATION_NAME}"
    COMPONENT "${APPLICATION_NAME}-py"
  )
endfunction()
