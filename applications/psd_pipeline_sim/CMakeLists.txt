# SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.20)
project(psd_pipeline_sim CXX CUDA)

find_package(holoscan 2.5.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

set(CMAKE_CUDA_ARCHITECTURES "70;80;90")
enable_language(CUDA)

add_executable(psd_pipeline_sim
  main.cpp
)

# Download MatX
include(FetchContent)
FetchContent_Declare(
  MatX
  GIT_REPOSITORY https://github.com/NVIDIA/MatX.git
  GIT_TAG v0.9.0
)
FetchContent_MakeAvailable(MatX)

target_link_libraries(psd_pipeline_sim PRIVATE
  matx::matx
  holoscan::core
  holoscan::ops::file_reader
  holoscan::ops::fft
  holoscan::ops::high_rate_psd
  holoscan::ops::low_rate_psd
  holoscan::ops::vita49_psd_packetizer
)
target_include_directories(psd_pipeline_sim PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
)

add_custom_target(config_yaml
    COMMAND ${CMAKE_COMMAND} -E copy ${CMAKE_CURRENT_SOURCE_DIR}/config.yaml ${CMAKE_CURRENT_BINARY_DIR}/config.yaml
)

add_dependencies(psd_pipeline_sim config_yaml)

if(BUILD_TESTING)
  # Generate some random data
  add_custom_target(rand_data
      COMMAND dd if=/dev/urandom of=${CMAKE_CURRENT_BINARY_DIR}/rand_data.dat bs=32000 count=1000
  )
  add_dependencies(psd_pipeline_sim rand_data)

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml" CONFIG_FILE)
  string(REGEX REPLACE "num_psds:[^\n]*" "num_psds: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REGEX REPLACE "file_name:[^\n]*" "file_name: rand_data.dat" CONFIG_FILE ${CONFIG_FILE})
  file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/config_testing.yaml ${CONFIG_FILE})
  add_test(NAME psd_pipeline_sim_test
           COMMAND psd_pipeline_sim config_testing.yaml
           WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

  # Make sure we get properly-sized PSD packet out the end of the pipeline
  set_tests_properties(psd_pipeline_sim_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Sending 1280 sample spectral data packet to 127.0.0.1:4991/udp"
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")
endif()
