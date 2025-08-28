# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)
project(streaming_client_operator)

# Find Holoscan - use simple approach
find_package(holoscan 3.2 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Add the current directory to CMAKE_MODULE_PATH so we can find FindHoloscanStreaming.cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}")

# Find HoloscanStreaming using our custom module
find_package(HoloscanStreaming REQUIRED)

message(STATUS "StreamingClient operator using HoloscanStreaming imported targets")

# Define the operator as a library
add_library(${PROJECT_NAME}
  SHARED
  streaming_client.cpp
)

add_library(holoscan::ops::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# Set C++ standard
set_target_properties(${PROJECT_NAME} PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# Set up include directories
target_include_directories(${PROJECT_NAME}
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against HoloscanStreaming and Holoscan using imported targets
target_link_libraries(${PROJECT_NAME}
  PUBLIC
  HoloscanStreaming::All
  holoscan::core
  PRIVATE
  dl
  rt
  atomic
)

# Add linking options to ensure all symbols are resolved
target_link_options(${PROJECT_NAME} PRIVATE
  "-Wl,--no-as-needed"
  "-Wl,--allow-shlib-undefined"
  "-Wl,--copy-dt-needed-entries"
  "-Wl,--disable-new-dtags"
)

# Set RPATH settings - simplified to avoid circular dependencies
set_target_properties(${PROJECT_NAME} PROPERTIES
  SKIP_BUILD_RPATH FALSE
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "$ORIGIN:$ORIGIN/lib:/usr/lib/x86_64-linux-gnu:${CMAKE_INSTALL_PREFIX}/lib:/opt/nvidia/holoscan/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Copy HoloscanStreaming libraries to the build directory using helper function
copy_holoscan_streaming_libraries(${PROJECT_NAME} "${CMAKE_CURRENT_BINARY_DIR}/lib")

# Copy the operator library to the application directory
add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_BINARY_DIR}/operators/streaming_client
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  $<TARGET_FILE:${PROJECT_NAME}>
  ${CMAKE_BINARY_DIR}/operators/streaming_client/lib${PROJECT_NAME}.so
)

# Install the operator shared library
install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}-targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install headers
install(
  FILES
    streaming_client.hpp
    include/StreamingClient.h
    include/VideoFrame.h
  DESTINATION
    include
)

# Install all HoloscanStreaming libraries
install(
  DIRECTORY ${HoloscanStreaming_LIB_DIR}/
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*"
)

# Export the targets for use by other projects
install(EXPORT ${PROJECT_NAME}-targets
  FILE ${PROJECT_NAME}-targets.cmake
  NAMESPACE holoscan::ops::
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# Create a config file for the package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  VERSION 1.0.0
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/${PROJECT_NAME}-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  INSTALL_DESTINATION lib/cmake/${PROJECT_NAME}
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}-config-version.cmake"
  DESTINATION lib/cmake/${PROJECT_NAME}
)

# Python module
if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# Add tests subdirectory
if(BUILD_TESTING)
  add_subdirectory(test)
endif()

# Print status message
message(STATUS "StreamingClient operator configuration complete with imported targets.")
