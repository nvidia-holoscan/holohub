# SPDX-FileCopyrightText: Copyright (c) 2026, Chris von Csefalvay.
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.20)
project(foxglove_endoscopy_tool_tracking CXX)

find_package(holoscan 4.1 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)

add_executable(foxglove_endoscopy_tool_tracking
  main.cpp
)

target_link_libraries(foxglove_endoscopy_tool_tracking
  PRIVATE
    holoscan::core
    holoscan::ops::video_stream_replayer
    holoscan::ops::format_converter
    holoscan::ops::foxglove
    lstm_tensor_rt_inference
    tool_tracking_postprocessor
    CUDA::cudart
)

if(HOLOHUB_DOWNLOAD_DATASETS)
  add_dependencies(foxglove_endoscopy_tool_tracking endoscopy_data)
endif()

add_custom_target(foxglove_endoscopy_tool_tracking_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${CMAKE_CURRENT_SOURCE_DIR}/foxglove_endoscopy_tool_tracking.yaml"
          "${CMAKE_CURRENT_BINARY_DIR}"
  DEPENDS "foxglove_endoscopy_tool_tracking.yaml"
  BYPRODUCTS "foxglove_endoscopy_tool_tracking.yaml"
)
add_dependencies(foxglove_endoscopy_tool_tracking foxglove_endoscopy_tool_tracking_deps)

install(
  TARGETS foxglove_endoscopy_tool_tracking
  DESTINATION bin/foxglove_endoscopy_tool_tracking/cpp
)

install(
  FILES foxglove_endoscopy_tool_tracking.yaml
  DESTINATION bin/foxglove_endoscopy_tool_tracking/cpp
)
