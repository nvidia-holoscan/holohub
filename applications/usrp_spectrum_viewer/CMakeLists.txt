# SPDX-FileCopyrightText: 2026 National Instruments Corporation
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.24)
project(usrp_spectrum_viewer CXX CUDA)

# Dependencies
find_package(holoscan 4.4.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")
find_package(CUDAToolkit REQUIRED)
if(DEFINED ENV{DAQIRI_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "$ENV{DAQIRI_ROOT}")
endif()
list(PREPEND CMAKE_PREFIX_PATH "/opt/daqiri")
find_package(daqiri REQUIRED CONFIG)
include(HoloHubDaqiri)

enable_language(CUDA)

# Custom operator dependency
add_subdirectory(usrp_rx)
add_subdirectory(uhd_chdr_rx)
add_subdirectory(log)
add_subdirectory(spectrum_overlay)
add_subdirectory(spectrum_visualizer)
add_subdirectory(spectrum_magnitude)

# Create the executable
add_executable(usrp_spectrum_viewer
  main.cpp
)
target_link_libraries(usrp_spectrum_viewer PRIVATE
  holoscan::matx
  holoscan::core
  holoscan::ops::fft
  holoscan::ops::holoviz
  holoscan::viz::imgui
  usrp_rx
  uhd_chdr_rx
  log
  spectrum_overlay
  spectrum_magnitude
  spectrum_visualizer
)
holohub_link_daqiri(usrp_spectrum_viewer)

# Copy config file
add_custom_target(usrp_spectrum_viewer_config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml"
)
add_dependencies(usrp_spectrum_viewer usrp_spectrum_viewer_config_yaml)
