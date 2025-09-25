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

# Define path for StreamingServer dependencies (using lib directory)
set(STREAMING_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")

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
  INSTALL_RPATH "$ORIGIN/lib:${CMAKE_INSTALL_PREFIX}/lib:/opt/nvidia/holoscan/lib"
  INSTALL_RPATH_USE_LINK_PATH TRUE
  SKIP_BUILD_RPATH FALSE
)

# Add include directories
target_include_directories(streaming_server
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Note: Removed global link_directories to avoid RPATH conflicts
# Libraries are linked using full paths instead

# Link against libraries (use full paths for reliability)
target_link_libraries(streaming_server
  PUBLIC
    holoscan::core
    "${STREAMING_LIB_DIR}/libNvStreamBase.so"
    "${STREAMING_LIB_DIR}/libNvStreamServer.so"
    "${STREAMING_LIB_DIR}/libNvStreamingSession.so"
    "${STREAMING_LIB_DIR}/libPoco.so"
    "${STREAMING_LIB_DIR}/libcudart.so.12"
    "${STREAMING_LIB_DIR}/libssl.so.3"
    "${STREAMING_LIB_DIR}/libcrypto.so.3"
    "${STREAMING_LIB_DIR}/libStreamingServer.so"
    atomic
)

# List library filenames (only files that actually exist in NGC package)
set(STREAMING_SO_FILENAMES
    "libNvStreamBase.so"
    "libNvStreamServer.so"
    "libNvStreamingSession.so"
    "libPoco.so"
    "libcudart.so.12"
    "libcudart.so.12.0.107"
    "libssl.so.3"
    "libcrypto.so.3"
    "libStreamingServer.so"
)

# List plugin filenames (from plugins directory)
set(STREAMING_PLUGIN_FILENAMES
    "libNicllsHandlerServer.so"
    "libNvcfBlasUpload.so"
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

# Copy plugin files
add_custom_command(TARGET streaming_server PRE_LINK
  COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/lib/plugins"
)
foreach(PLUGIN_FILENAME ${STREAMING_PLUGIN_FILENAMES})
  add_custom_command(TARGET streaming_server PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${STREAMING_LIB_DIR}/plugins/${PLUGIN_FILENAME}"
      "${CMAKE_CURRENT_BINARY_DIR}/lib/plugins/"
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

# Copy plugin files to build directory
add_custom_command(TARGET streaming_server POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E make_directory
    ${CMAKE_BINARY_DIR}/operators/streaming_server/lib/plugins
)
foreach(PLUGIN_FILENAME ${STREAMING_PLUGIN_FILENAMES})
  add_custom_command(TARGET streaming_server POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
      "${STREAMING_LIB_DIR}/plugins/${PLUGIN_FILENAME}"
      "${CMAKE_BINARY_DIR}/operators/streaming_server/lib/plugins/"
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
    include/StreamingServer.h
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

# Python module
if(HOLOHUB_BUILD_PYTHON)
  add_subdirectory(python)
endif()

# Download test data for streaming server tests
# Default to download datasets for testing
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the endoscopy sample data for testing
if(HOLOHUB_DOWNLOAD_DATASETS AND BUILD_TESTING)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL nvidia/clara-holoscan/holoscan_endoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    DOWNLOAD_DIR ${CMAKE_BINARY_DIR}/data
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 480
    GXF_ENTITIES_WIDTH 854
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
  )
  # Create a symlink to make the data easily accessible at the expected location
  add_custom_command(TARGET endoscopy_data POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_BINARY_DIR}/data/endoscopy/surgical_video.gxf_entities
            ${CMAKE_BINARY_DIR}/data/surgical_video.gxf_entities
    COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_BINARY_DIR}/data/endoscopy/surgical_video.gxf_index
            ${CMAKE_BINARY_DIR}/data/surgical_video.gxf_index
    COMMENT "Creating symlinks for surgical video data"
  )
endif()

# Testing infrastructure
include(CTest)
if(BUILD_TESTING)
  # Register individual pytest tests with CTest using HoloHub's standard method
  find_package(Python3 COMPONENTS Interpreter)
  if(Python3_FOUND)
    # Check if pytest is available
    execute_process(
      COMMAND ${Python3_EXECUTABLE} -c "import pytest"
      RESULT_VARIABLE PYTEST_AVAILABLE
      OUTPUT_QUIET ERROR_QUIET
    )
    # Always use manual pytest test registration for reliability
    message(STATUS "Adding pytest tests manually...")
    add_test(NAME streaming_server_operator_unit_test
             COMMAND /bin/bash -c "${Python3_EXECUTABLE} -m pytest tests/test_streaming_server_op.py -v"
             WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
    set_tests_properties(streaming_server_operator_unit_test PROPERTIES
                         ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
                         TIMEOUT 120
                         PASS_REGULAR_EXPRESSION "16 passed"
                         FAIL_REGULAR_EXPRESSION "failed|FAILED|ERROR")

    # Set up data download fixture if enabled
    if(HOLOHUB_DOWNLOAD_DATASETS AND TARGET endoscopy_data)
      set_tests_properties(streaming_server_operator_unit_test PROPERTIES
                           FIXTURES_SETUP endoscopy_data_downloaded)
    endif()
  endif()

  # Infrastructure tests (basic functionality without video data)
  add_test(NAME streaming_server_operator_infrastructure_test
           COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_test.sh"
                   "${CMAKE_BINARY_DIR}/applications/video_streaming/video_streaming_server/python"
                   "${CMAKE_BINARY_DIR}/applications/video_streaming/video_streaming_server/python/streaming_server_demo.py"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  set_tests_properties(streaming_server_operator_infrastructure_test PROPERTIES
                       ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
                       TIMEOUT 120
                       PASS_REGULAR_EXPRESSION "StreamingServer functionality validated successfully"
                       FAIL_REGULAR_EXPRESSION "Test FAILED:")

  # Functional tests (with real video data, if available)
  if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/video_streaming_server_functional.py")
      add_test(NAME streaming_server_operator_functional_test
               COMMAND bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_functional_test.sh"
                       "${CMAKE_BINARY_DIR}/operators/streaming_server/python"
                       "${CMAKE_CURRENT_SOURCE_DIR}/tests/video_streaming_server_functional.py"
                       "${CMAKE_BINARY_DIR}/data/endoscopy"
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    set_tests_properties(streaming_server_operator_functional_test PROPERTIES
                         ENVIRONMENT "PYTHONPATH=${CMAKE_BINARY_DIR}/python/lib:$ENV{PYTHONPATH}"
                         TIMEOUT 120
                         PASS_REGULAR_EXPRESSION "FUNCTIONAL test PASSED.*Python StreamingServer.*successful"
                         FAIL_REGULAR_EXPRESSION "FUNCTIONAL test.*FAILED")

    # Make functional tests depend on data download if data downloading is enabled
    if(HOLOHUB_DOWNLOAD_DATASETS AND TARGET endoscopy_data)
      set_tests_properties(streaming_server_operator_functional_test PROPERTIES
                           FIXTURES_REQUIRED endoscopy_data_downloaded)
    endif()
  endif()

  # C++ tests (if streaming_server_demo target exists)
  if(TARGET streaming_server_demo)
    add_test(NAME streaming_server_operator_cpp_test
             COMMAND ${CMAKE_COMMAND} -E env
                     "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/operators/streaming_server:${CMAKE_BINARY_DIR}/operators/streaming_server/lib:${CMAKE_SOURCE_DIR}/operators/streaming_server/lib:${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
                     bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_cpp_test.sh"
                     "$<TARGET_FILE:streaming_server_demo>"
                     "${CMAKE_CURRENT_SOURCE_DIR}/testing/video_streaming_server_testing.yaml"
                     "${CMAKE_BINARY_DIR}/data/endoscopy"
             WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

    set_tests_properties(streaming_server_operator_cpp_test PROPERTIES
                         TIMEOUT 120
                         PASS_REGULAR_EXPRESSION "C\\+\\+ StreamingServer.*test successful"
                         FAIL_REGULAR_EXPRESSION "Test FAILED:")

    # Make C++ tests depend on data download if data downloading is enabled
    if(HOLOHUB_DOWNLOAD_DATASETS AND TARGET endoscopy_data)
      set_tests_properties(streaming_server_operator_cpp_test PROPERTIES
                           FIXTURES_REQUIRED endoscopy_data_downloaded)
      add_dependencies(streaming_server_demo endoscopy_data)
    endif()

    # C++ functional test (if script exists)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_cpp_functional_test.sh")
      add_test(NAME streaming_server_operator_cpp_functional_test
               COMMAND ${CMAKE_COMMAND} -E env
                       "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/operators/streaming_server:${CMAKE_BINARY_DIR}/operators/streaming_server/lib:${CMAKE_SOURCE_DIR}/operators/streaming_server/lib:${CMAKE_BINARY_DIR}/lib:$ENV{LD_LIBRARY_PATH}"
                       bash "${CMAKE_CURRENT_SOURCE_DIR}/tests/run_cpp_functional_test.sh"
                       "$<TARGET_FILE:streaming_server_demo>"
                       "${CMAKE_CURRENT_SOURCE_DIR}/testing/video_streaming_server_testing.yaml"
                       "${CMAKE_BINARY_DIR}/data/endoscopy"
               WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

      set_tests_properties(streaming_server_operator_cpp_functional_test PROPERTIES
                           TIMEOUT 120
                           PASS_REGULAR_EXPRESSION "FUNCTIONAL test PASSED.*StreamingServer.*successful"
                           FAIL_REGULAR_EXPRESSION "FUNCTIONAL test.*FAILED")

      # Make C++ functional test depend on data download if data downloading is enabled
      if(HOLOHUB_DOWNLOAD_DATASETS AND TARGET endoscopy_data)
        set_tests_properties(streaming_server_operator_cpp_functional_test PROPERTIES
                             FIXTURES_REQUIRED endoscopy_data_downloaded)
      endif()
    endif()
  endif()
endif()

# Print status message
message(STATUS "StreamingServer operator configuration complete.")
