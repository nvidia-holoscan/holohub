# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(streaming_server_demo CXX)

# Add executable
add_executable(streaming_server_demo
    streaming_server_demo.cpp
)

# Print success message
message(STATUS "✓ Executable target 'streaming_server_demo' created")

# Set build directory
message(STATUS "Build streaming_server_demo directory: ${CMAKE_BINARY_DIR}/streaming_server_demo")

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Set compile definitions
target_compile_definitions(streaming_server_demo
    PRIVATE
    HOLOSCAN_MAJOR_VERSION=${holoscan_VERSION_MAJOR}
    HOLOSCAN_MINOR_VERSION=${holoscan_VERSION_MINOR}
    HOLOSCAN_PATCH_VERSION=${holoscan_VERSION_PATCH}
)

# Set compile options
target_compile_options(streaming_server_demo
    PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall>
    $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-reorder>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
    $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-compare>
)

# Add dependency on streaming_server operator
add_dependencies(streaming_server_demo streaming_server)

# Link libraries
target_link_libraries(streaming_server_demo
    PUBLIC
    holoscan::core
    holoscan::streaming_server
    CUDA::cudart
    CUDA::cuda_driver
)



# Set target properties
set_target_properties(streaming_server_demo PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SKIP_BUILD_RPATH FALSE
)

# Add include directories
target_include_directories(streaming_server_demo
    PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# Copy YAML file if it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml")
    add_custom_command(TARGET streaming_server_demo POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
        "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml"
        "$<TARGET_FILE_DIR:streaming_server_demo>/streaming_server_demo.yaml"
    )
endif()

# Create lib directory
add_custom_command(TARGET streaming_server_demo POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:streaming_server_demo>/lib"
)

# Install rules
install(
    TARGETS streaming_server_demo
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)

# Install config files
install(
    FILES
    "${CMAKE_CURRENT_SOURCE_DIR}/streaming_server_demo.yaml"
    DESTINATION bin
)

# Comprehensive testing infrastructure following streaming_client_demo pattern
if(BUILD_TESTING)
  # Ensure data dependencies are available for testing
  if(HOLOHUB_DOWNLOAD_DATASETS)
    include(holoscan_download_data)
    holoscan_download_data(endoscopy
      URL nvidia/clara-holoscan/holoscan_endoscopy_sample_data:20230222
      DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
      DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
      GENERATE_GXF_ENTITIES
      GXF_ENTITIES_HEIGHT 480
      GXF_ENTITIES_WIDTH 854
      GXF_ENTITIES_CHANNELS 3
      GXF_ENTITIES_FRAMERATE 30
      ALL
    )
    add_dependencies(streaming_server_demo endoscopy_data)
  endif()

  # Configure comprehensive testing directories
  set(RECORDING_DIR "${CMAKE_CURRENT_BINARY_DIR}/recording_output")
  set(SOURCE_VIDEO_BASENAME "streaming_server_demo_reference")
  set(VALIDATION_FRAMES_DIR "${CMAKE_SOURCE_DIR}/applications/streaming_server_demo/testing")
  
  # Ensure all testing directories exist
  file(MAKE_DIRECTORY ${RECORDING_DIR})
  file(MAKE_DIRECTORY ${VALIDATION_FRAMES_DIR})

  # Copy and update the test configuration
  configure_file(
    "${CMAKE_SOURCE_DIR}/applications/streaming_server_demo/testing/streaming_server_demo_testing.yaml"
    "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_testing.yaml"
    COPYONLY
  )

  # Read and modify the test configuration for specific test parameters
  file(READ "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_testing.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "directory: \"./recording_output\"" "directory: \"${RECORDING_DIR}\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "basename: \"streaming_server_demo_reference\"" "basename: \"${SOURCE_VIDEO_BASENAME}\"" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_testing.yaml" ${CONFIG_FILE})

  # Add comprehensive C++ functional test
  add_test(NAME streaming_server_demo_cpp_test
           COMMAND bash ${CMAKE_CURRENT_SOURCE_DIR}/run_test.sh
                   ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo
                   ${CMAKE_CURRENT_BINARY_DIR}/streaming_server_demo_testing.yaml
                   ${HOLOHUB_DATA_DIR}/endoscopy
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  
  # Set comprehensive test properties with enhanced environment and validation
  set_tests_properties(streaming_server_demo_cpp_test PROPERTIES
                       ENVIRONMENT "LD_LIBRARY_PATH=${CMAKE_BINARY_DIR}/operators/streaming_server:${CMAKE_BINARY_DIR}/operators/streaming_server/lib:${CMAKE_SOURCE_DIR}/operators/streaming_server/lib:$ENV{LD_LIBRARY_PATH}"
                       TIMEOUT 120
                       PASS_REGULAR_EXPRESSION "Test PASSED: C\\+\\+ StreamingServer.*test successful"
                       FAIL_REGULAR_EXPRESSION "Test FAILED:")

  # Video validation test disabled for streaming server - server doesn't generate 
  # the expected GXF output format for validation. Server testing focuses on
  # functionality and unit testing rather than visual output validation.
  # TODO: Re-enable when server output recording is implemented
  # add_test(NAME streaming_server_demo_cpp_render_test
  #     COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
  #     --source_video_dir ${VALIDATION_FRAMES_DIR}
  #     --source_video_basename "streaming_server_demo_reference"
  #     --output_dir ${RECORDING_DIR}
  #     --validation_frames_dir ${VALIDATION_FRAMES_DIR}
  #     WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  # )
  # set_tests_properties(streaming_server_demo_cpp_render_test PROPERTIES
  #     DEPENDS streaming_server_demo_cpp_test
  #     PASS_REGULAR_EXPRESSION "Valid video output!"
  # )

endif()

