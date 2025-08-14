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
project(streaming_server)

# Find Holoscan package
find_package(holoscan 3.2 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Define path for StreamingServer dependencies (downloaded to build tree)
set(STREAMING_LIB_DIR "${CMAKE_CURRENT_BINARY_DIR}/streaming_libs")

# Print library directory
message(STATUS "Using Streaming library directory: ${STREAMING_LIB_DIR}")

# Set ABI flag consistently to match StreamingServer library
add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)

# Add additional compiler and linker flags for stability and security
list(APPEND CMAKE_CXX_FLAGS "-fstack-protector-strong")
list(APPEND CMAKE_SHARED_LINKER_FLAGS "-Wl,-z,relro,-z,now")

# Define the operator as a library
add_library(streaming_server
  SHARED
  streaming_server_op.cpp
)

add_library(holoscan::streaming_server ALIAS streaming_server)

# Set C++ standard and other properties
set_target_properties(streaming_server PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
  BUILD_WITH_INSTALL_RPATH TRUE
  INSTALL_RPATH "${STREAMING_LIB_DIR}:$ORIGIN/lib:${CMAKE_INSTALL_PREFIX}/lib:/opt/nvidia/holoscan/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
  SKIP_BUILD_RPATH FALSE
)

# Add include directories
target_include_directories(streaming_server
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Add link directory explicitly
link_directories(${STREAMING_LIB_DIR})

# Link against libraries (paths resolved via link_directories)
target_link_libraries(streaming_server
  PUBLIC
    holoscan::core
    NvStreamBase
    NvStreamServer
    NvStreamingSession
    "${STREAMING_LIB_DIR}/libNvStreamingMedia.a"  # Static lib needs full path
    StreamClientShared
    SCI
    "protobuf.so.9"  # Versioned lib needs explicit name
    Poco
    NvSignalingServer
    NvSessionNegotiator
    "${STREAMING_LIB_DIR}/libNVCodec.a"  # Static lib needs full path
    NvcfBlasUpload
    NicllsHandlerServer
    messagebus
    InputStreamShared
    "cudart.so.12"  # Versioned lib needs explicit name
    "cudart.so.12.0.107"  # Specific version needs explicit name
    "ssl.so.3"  # Versioned lib needs explicit name
    "crypto.so.3"  # Versioned lib needs explicit name
    AudioStreamShared
    serverlib
    atomic
)

# List library filenames (paths constructed in foreach loop)
set(STREAMING_SO_FILENAMES
    "libNvStreamBase.so"
    "libNvStreamServer.so"
    "libNvStreamingSession.so"
    "libStreamClientShared.so"
    "libSCI.so"
    "libprotobuf.so.9"
    "libPoco.so"
    "libNvSignalingServer.so"
    "libNvSessionNegotiator.so"
    "libNvcfBlasUpload.so"
    "libNicllsHandlerServer.so"
    "libmessagebus.so"
    "libInputStreamShared.so"
    "libcudart.so.12"
    "libcudart.so.12.0.107"
    "libssl.so.3"
    "libcrypto.so.3"
    "libAudioStreamShared.so"
    "libserverlib.so"
)

# Make sure all StreamingServer libraries are copied to the build directory first
add_custom_command(TARGET streaming_server PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/lib"
)

# Copy all .so files (construct full paths from filenames)
foreach(SO_FILENAME ${STREAMING_SO_FILENAMES})
  add_custom_command(TARGET streaming_server PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${STREAMING_LIB_DIR}/${SO_FILENAME}"
      "${CMAKE_CURRENT_BINARY_DIR}/lib/"
  )
endforeach()

# Copy executables (like server)
foreach(EXEC_FILE ${STREAMING_EXECUTABLES})
  if(EXISTS "${EXEC_FILE}")
    add_custom_command(TARGET streaming_server PRE_LINK
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${EXEC_FILE}"
        "${CMAKE_CURRENT_BINARY_DIR}/lib/"
    )
  endif()
endforeach()

# Copy the operator library to the application directory
add_custom_command(TARGET streaming_server POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_BINARY_DIR}/operators/streaming_server
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
    $<TARGET_FILE:streaming_server>
    ${CMAKE_BINARY_DIR}/operators/streaming_server/libstreaming_server.so
)

# Also copy all .so files to the operator build directory
add_custom_command(TARGET streaming_server POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_BINARY_DIR}/operators/streaming_server/lib
)

foreach(SO_FILENAME ${STREAMING_SO_FILENAMES})
  add_custom_command(TARGET streaming_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${STREAMING_LIB_DIR}/${SO_FILENAME}"
      "${CMAKE_BINARY_DIR}/operators/streaming_server/lib/"
  )
endforeach()

foreach(EXEC_FILE ${STREAMING_EXECUTABLES})
  if(EXISTS "${EXEC_FILE}")
    add_custom_command(TARGET streaming_server POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${EXEC_FILE}"
        "${CMAKE_BINARY_DIR}/operators/streaming_server/lib/"
    )
  endif()
endforeach()

# Install the operator shared library and headers
install(TARGETS streaming_server
  EXPORT streaming_server-targets
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin
  INCLUDES DESTINATION include
)

install(
  FILES
    streaming_server_op.hpp
    StreamingServer.h
  DESTINATION
    include
)

# Install StreamingServer library and other dependencies
install(
  DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/
  DESTINATION lib
  FILES_MATCHING PATTERN "*.so*"
)

# Install executables
install(
  PROGRAMS ${STREAMING_EXECUTABLES}
  DESTINATION bin
)

# Export targets for find_package support
install(EXPORT streaming_server-targets
  FILE streaming_server-targets.cmake
  NAMESPACE holoscan::
  DESTINATION lib/cmake/streaming_server
)

# Create the cmake directory if it doesn't exist
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/cmake)

# Generate the config file
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/cmake/streaming_server-config.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/streaming_server-config.cmake
  @ONLY
)

# Create the config file template if it doesn't exist
if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/streaming_server-config.cmake.in)
  file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  file(WRITE ${CMAKE_CURRENT_SOURCE_DIR}/cmake/streaming_server-config.cmake.in
"@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
find_dependency(holoscan 3.2 REQUIRED)

include(\${CMAKE_CURRENT_LIST_DIR}/streaming_server-targets.cmake)
")
endif()

# Install the config file
install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/streaming_server-config.cmake
  DESTINATION
    lib/cmake/streaming_server
)

# Create package version file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/streaming_server-config-version.cmake
  VERSION 1.0.0
  COMPATIBILITY SameMajorVersion
)

install(
  FILES
    ${CMAKE_CURRENT_BINARY_DIR}/streaming_server-config-version.cmake
  DESTINATION
    lib/cmake/streaming_server
)

# Export package for use from build tree
export(EXPORT streaming_server-targets
  FILE ${CMAKE_CURRENT_BINARY_DIR}/cmake/streaming_server-targets.cmake
  NAMESPACE holoscan::
)

# Print status message
message(STATUS "StreamingServer operator configuration complete.")
