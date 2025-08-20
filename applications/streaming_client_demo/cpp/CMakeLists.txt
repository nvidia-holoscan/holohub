# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Set source file
set(SOURCE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.cpp")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(holoscan REQUIRED)

# Add executable
add_executable(streaming_client_demo
    streaming_client_demo.cpp
)



# Set compile definitions
target_compile_definitions(streaming_client_demo
    PRIVATE
    HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
    HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
    HOLOSCAN_PATCH_VERSION=${holoscan_VERSION_PATCH}
)

# Set compile features
target_compile_features(streaming_client_demo
    PUBLIC
    cxx_std_17
)

# Set compile options
target_compile_options(streaming_client_demo
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-reorder>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
)


# Link libraries
target_link_libraries(streaming_client_demo
    PUBLIC
    holoscan::core
    holoscan::ops::streaming_client_operator
    holoscan::ops::video_stream_replayer
    holoscan::ops::holoviz
    holoscan::ops::format_converter
    CUDA::cudart
    CUDA::cuda_driver
)

# Set target properties
set_target_properties(streaming_client_demo PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SKIP_BUILD_RPATH FALSE
)

# Add include directories
target_include_directories(streaming_client_demo
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Copy executable and config files
add_custom_command(TARGET streaming_client_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/bin"
)

# Copy YAML file if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml")
    add_custom_command(TARGET streaming_client_demo POST_BUILD
        # Copy to current binary dir
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo.yaml"

        # Copy to bin directory
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "${CMAKE_BINARY_DIR}/bin/streaming_client_demo.yaml"

        # Copy to build root
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "${CMAKE_BINARY_DIR}/streaming_client_demo.yaml"

        # Copy to build/streaming_client_demo directory (CMake creates directory automatically)
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "${CMAKE_BINARY_DIR}/streaming_client_demo/streaming_client_demo.yaml"

        # Copy to target directory
        COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" "$<TARGET_FILE_DIR:streaming_client_demo>/streaming_client_demo.yaml"
    )
endif()

# Copy executable to bin directory
add_custom_command(TARGET streaming_client_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:streaming_client_demo>" "${CMAKE_BINARY_DIR}/bin/"
)

# Install rules
install(
    TARGETS streaming_client_demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Install config files
install(
    FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml"
    DESTINATION bin
)

# Add testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME cpp_streaming_client_demo_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/streaming_client_demo/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/streaming_client_demo.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "record_type: \"none\"" "record_type: \"visualizer\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "directory: \"./recording_output\"" "directory: \"${RECORDING_DIR}\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "basename: \"streaming_client_demo_output\"" "basename: \"${SOURCE_VIDEO_BASENAME}\"" CONFIG_FILE ${CONFIG_FILE})
  
  # Set test mode parameters
  string(REPLACE "repeat: true" "repeat: false" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "realtime: true" "realtime: false" CONFIG_FILE ${CONFIG_FILE})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_testing.yaml" ${CONFIG_FILE})

  # Use real endoscopy video data for functional testing
  # The test system automatically provides real video data at ${HOLOHUB_DATA_DIR}/endoscopy
  # which contains surgical_video.gxf_index and surgical_video.gxf_entities with real frame data

  # Add test for application execution using wrapper script to handle exceptions gracefully
  add_test(NAME streaming_client_demo_cpp_test
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh
                   ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo
                   ${CMAKE_CURRENT_BINARY_DIR}/streaming_client_demo_testing.yaml
                   ${HOLOHUB_DATA_DIR}/endoscopy
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  
  # Set test properties including library path and pass/fail criteria  
  set_tests_properties(streaming_client_demo_cpp_test PROPERTIES
                       ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/operators/streaming_client:${CMAKE_BINARY_DIR}/operators/streaming_client/lib:${CMAKE_SOURCE_DIR}/operators/streaming_client/lib:$ENV{LD_LIBRARY_PATH}"
                       TIMEOUT 90
                       PASS_REGULAR_EXPRESSION "Test PASSED: C\\+\\+ StreamingClient functionality validated successfully"
                       FAIL_REGULAR_EXPRESSION "Test FAILED:")

  # Frame validation test disabled - current streaming client setup doesn't record output
  # TODO: Re-enable when recording functionality is implemented for streaming client

endif()


