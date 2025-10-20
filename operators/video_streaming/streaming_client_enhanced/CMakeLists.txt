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
project(streaming_client)

# Find Holoscan - use simple approach
find_package(holoscan 3.2 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Set the StreamingClient directory (following server pattern)
set(STREAMINGCLIENT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/holoscan_client_cloud_streaming)

# Detect architecture
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ARCH_DIR "x86_64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(ARCH_DIR "aarch64")
else()
    set(ARCH_DIR "x86_64") # Default to x86_64
endif()

# Find StreamingClient libraries directly (like server does)
find_library(StreamingClient_LIBRARY
    NAMES StreamingClient libStreamingClient
    PATHS ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}
    REQUIRED
)

find_library(StreamClientShared_LIBRARY
    NAMES StreamClientShared libStreamClientShared
    PATHS ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}
    REQUIRED
)

find_library(NvStreamBase_LIBRARY
    NAMES NvStreamBase libNvStreamBase
    PATHS ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}
    REQUIRED
)

find_library(NvStreamingSession_LIBRARY
    NAMES NvStreamingSession libNvStreamingSession
    PATHS ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}
    REQUIRED
)

find_library(Poco_LIBRARY
    NAMES Poco libPoco
    PATHS ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}
    REQUIRED
)

message(STATUS "StreamingClient operator using direct library linking")

# Define the operator as a library
add_library(streaming_client
  SHARED
  streaming_client.cpp
)

add_library(holoscan::ops::streaming_client_enhanced ALIAS streaming_client)

# Set C++ standard
set_target_properties(streaming_client PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)

# Set up include directories
target_include_directories(streaming_client
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${STREAMINGCLIENT_DIR}/include>
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

# Link against StreamingClient and Holoscan using direct libraries (like server does)
target_link_libraries(streaming_client
  PUBLIC
  holoscan::core
  ${StreamingClient_LIBRARY}
  ${StreamClientShared_LIBRARY}
  ${NvStreamBase_LIBRARY}
  ${NvStreamingSession_LIBRARY}
  ${Poco_LIBRARY}
  PRIVATE
  dl
  rt
  atomic
)

# Add linking options to ensure all symbols are resolved
target_link_options(streaming_client PRIVATE
  "-Wl,--no-as-needed"
  "-Wl,--allow-shlib-undefined"
  "-Wl,--copy-dt-needed-entries"
  "-Wl,--disable-new-dtags"
)

# Set RPATH settings - ensure all libraries can be found at runtime, system libs first
set_target_properties(streaming_client PROPERTIES
  SKIP_BUILD_RPATH FALSE
  BUILD_WITH_INSTALL_RPATH FALSE
  BUILD_RPATH "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu:${CMAKE_BINARY_DIR}/lib:${CMAKE_BINARY_DIR}/operators/streaming_client:${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}"
  INSTALL_RPATH "/usr/lib/${CMAKE_SYSTEM_PROCESSOR}-linux-gnu:$ORIGIN:$ORIGIN/lib:$ORIGIN/../lib:$ORIGIN/../../lib:${CMAKE_INSTALL_PREFIX}/lib:/opt/nvidia/holoscan/lib:${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}"
  INSTALL_RPATH_USE_LINK_PATH TRUE
)

# Copy the operator library to the application directory and ensure ALL dependencies are available
add_custom_command(TARGET streaming_client POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_BINARY_DIR}/operators/streaming_client
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  $<TARGET_FILE:streaming_client>
  ${CMAKE_BINARY_DIR}/operators/streaming_client/libstreaming_client.so
  # Copy ALL required streaming libraries to ensure runtime availability
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libStreamingClient.so
  ${CMAKE_BINARY_DIR}/operators/streaming_client/libStreamingClient.so
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libNvStreamingSession.so
  ${CMAKE_BINARY_DIR}/operators/streaming_client/libNvStreamingSession.so
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libNvStreamBase.so
  ${CMAKE_BINARY_DIR}/operators/streaming_client/libNvStreamBase.so
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libStreamClientShared.so
  ${CMAKE_BINARY_DIR}/operators/streaming_client/libStreamClientShared.so
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libPoco.so
  ${CMAKE_BINARY_DIR}/operators/streaming_client/libPoco.so
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libssl.so.3
  ${CMAKE_BINARY_DIR}/operators/streaming_client/libssl.so.3
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libcrypto.so.3
  ${CMAKE_BINARY_DIR}/operators/streaming_client/libcrypto.so.3
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libcudart.so.12.0.107
  ${CMAKE_BINARY_DIR}/operators/streaming_client/libcudart.so.12.0.107
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libcudart.so.12
  ${CMAKE_BINARY_DIR}/operators/streaming_client/libcudart.so.12
  # Copy libraries to the main lib directory as well
  COMMAND ${CMAKE_COMMAND} -E make_directory
  ${CMAKE_BINARY_DIR}/lib
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libNvStreamingSession.so
  ${CMAKE_BINARY_DIR}/lib/libNvStreamingSession.so
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
  ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libStreamingClient.so
  ${CMAKE_BINARY_DIR}/lib/libStreamingClient.so
)

# Install the operator shared library
install(TARGETS streaming_client
  EXPORT streaming_client-targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

# Install headers
install(
  FILES
    streaming_client.hpp
    ${STREAMINGCLIENT_DIR}/include/StreamingClient.h
    ${STREAMINGCLIENT_DIR}/include/VideoFrame.h
  DESTINATION
    include
)

# Explicitly install each required StreamingClient library - CMake will fail if any are missing
# Core StreamingClient library
install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libStreamingClient.so
  DESTINATION lib
)

# NVIDIA Streaming Session library (critical for functionality)
install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libNvStreamingSession.so
  DESTINATION lib
)

# NVIDIA Stream Base library
install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libNvStreamBase.so
  DESTINATION lib
)

# Stream Client Shared library
install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libStreamClientShared.so
  DESTINATION lib
)

# Poco library
install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libPoco.so
  DESTINATION lib
)

# SSL/Crypto libraries
install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libssl.so.3
  DESTINATION lib
)

install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libcrypto.so.3
  DESTINATION lib
)

# CUDA runtime libraries
install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libcudart.so.12.0.107
  DESTINATION lib
)

install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libcudart.so.12
  DESTINATION lib
)

# Install symlinks if they exist
if(EXISTS ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libssl.so)
  install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libssl.so
    DESTINATION lib
  )
endif()

if(EXISTS ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libcrypto.so)
  install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libcrypto.so
    DESTINATION lib
  )
endif()

if(EXISTS ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libcudart.so)
  install(FILES ${STREAMINGCLIENT_DIR}/lib/${ARCH_DIR}/libcudart.so
    DESTINATION lib
  )
endif()

# Export the targets for use by other projects
install(EXPORT streaming_client-targets
  FILE streaming_client-targets.cmake
  NAMESPACE holoscan::ops::
  DESTINATION lib/cmake/streaming_client
)

# Create a config file for the package
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/streaming_client-config-version.cmake"
  VERSION 1.0.0
  COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client-config.cmake.in"
  "${CMAKE_CURRENT_BINARY_DIR}/streaming_client-config.cmake"
  INSTALL_DESTINATION lib/cmake/streaming_client
)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/streaming_client-config.cmake"
  "${CMAKE_CURRENT_BINARY_DIR}/streaming_client-config-version.cmake"
  "${CMAKE_CURRENT_SOURCE_DIR}/FindStreamingClient.cmake"
  DESTINATION lib/cmake/streaming_client
)


# Print status message
message(STATUS "StreamingClient operator configuration complete with imported targets.")
