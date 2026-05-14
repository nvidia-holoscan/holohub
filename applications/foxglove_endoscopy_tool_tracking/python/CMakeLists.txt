# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

add_custom_target(foxglove_endoscopy_tool_tracking_python ALL
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/foxglove_endoscopy_tool_tracking.py"
          "${CMAKE_CURRENT_BINARY_DIR}/foxglove_endoscopy_tool_tracking.py"
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/foxglove_endoscopy_tool_tracking.yaml"
          "${CMAKE_CURRENT_BINARY_DIR}/foxglove_endoscopy_tool_tracking.yaml"
  DEPENDS
    "foxglove_endoscopy_tool_tracking.py"
    "foxglove_endoscopy_tool_tracking.yaml"
)

install(
  FILES
    foxglove_endoscopy_tool_tracking.py
    foxglove_endoscopy_tool_tracking.yaml
  DESTINATION bin/foxglove_endoscopy_tool_tracking/python
)
