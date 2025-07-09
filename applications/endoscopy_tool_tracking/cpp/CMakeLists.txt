# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(endoscopy_tool_tracking CXX)

find_package(holoscan 1.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(endoscopy_tool_tracking
  main.cpp
)

target_link_libraries(endoscopy_tool_tracking
  PRIVATE
  holoscan::core
  holoscan::ops::video_stream_replayer
  holoscan::ops::video_stream_recorder
  holoscan::ops::format_converter
  holoscan::ops::holoviz
  lstm_tensor_rt_inference
  tool_tracking_postprocessor
)

target_link_libraries(endoscopy_tool_tracking PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::aja>)
target_link_libraries(endoscopy_tool_tracking PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::videomaster>)
target_link_libraries(endoscopy_tool_tracking PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::qcap_source>)
target_link_libraries(endoscopy_tool_tracking PRIVATE $<TARGET_NAME_IF_EXISTS:holoscan::vtk_renderer>)

# Download the associated dataset if needed
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)
if(HOLOHUB_DOWNLOAD_DATASETS)
  add_dependencies(endoscopy_tool_tracking endoscopy_data)
endif()

# Copy the config to the binary directory
add_custom_target(endoscopy_tool_tracking_deps
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy_if_different "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking_aja_overlay.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "endoscopy_tool_tracking.yaml" "endoscopy_tool_tracking_aja_overlay.yaml"
  BYPRODUCTS "endoscopy_tool_tracking.yaml" "endoscopy_tool_tracking_aja_overlay.yaml"
)
add_dependencies(endoscopy_tool_tracking endoscopy_tool_tracking_deps)

# Add testing
if(BUILD_TESTING)

  set(RECORDING_DIR ${CMAKE_CURRENT_BINARY_DIR}/recording_output)
  set(SOURCE_VIDEO_BASENAME cpp_endoscopy_tool_tracking_output)
  set(VALIDATION_FRAMES_DIR ${CMAKE_SOURCE_DIR}/applications/endoscopy_tool_tracking/testing/)

  file(MAKE_DIRECTORY ${RECORDING_DIR})

  # Configure the yaml file for testing
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/endoscopy_tool_tracking.yaml" CONFIG_FILE)
  string(REPLACE "count: 0" "count: 10" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "record_type: \"none\"" "record_type: \"visualizer\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "directory: \"/tmp\"" "directory: \"${RECORDING_DIR}\"" CONFIG_FILE ${CONFIG_FILE})
  string(REPLACE "basename: \"tensor\"" "basename: \"${SOURCE_VIDEO_BASENAME}\"" CONFIG_FILE ${CONFIG_FILE})

  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml" ${CONFIG_FILE})

  # Add test
  add_test(NAME endoscopy_tool_tracking_cpp_test
           COMMAND endoscopy_tool_tracking
                   --config ${CMAKE_CURRENT_BINARY_DIR}/endoscopy_tool_tracking_testing.yaml
                   --data "${HOLOHUB_DATA_DIR}/endoscopy"
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  set_tests_properties(endoscopy_tool_tracking_cpp_test PROPERTIES
                       PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
                       FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed")

  # Add a test to check the validity of the frames
  add_test(NAME endoscopy_tool_tracking_cpp_render_test
      COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/video_validation.py
      --source_video_dir ${RECORDING_DIR}
      --source_video_basename ${SOURCE_VIDEO_BASENAME}
      --output_dir ${RECORDING_DIR}
      --validation_frames_dir ${VALIDATION_FRAMES_DIR}
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )

  set_tests_properties(endoscopy_tool_tracking_cpp_render_test PROPERTIES
  DEPENDS endoscopy_tool_tracking_cpp_test
  PASS_REGULAR_EXPRESSION "Valid video output!"
  )

endif()

# Flow benchmarking
if(FLOW_BENCHMARKING)
  # Patch the current source file
  file(READ "${CMAKE_CURRENT_SOURCE_DIR}/main.cpp" MAIN_CPP_FILE)
  string(REGEX REPLACE "^(.*holoscan::Application.*)" "#include \"benchmark.hpp\"\\n\\1" MAIN_CPP_FILE "${MAIN_CPP_FILE}")
  string(REPLACE "holoscan::Application" "BenchmarkedApplication" MAIN_CPP_FILE "${MAIN_CPP_FILE}")
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/main_benchmark.cpp" "${MAIN_CPP_FILE}")

  # Create a new executable
  add_executable(endoscopy_tool_tracking_flowbenchmark ${CMAKE_CURRENT_BINARY_DIR}/main_benchmark.cpp)
  get_target_property(ENDOSCOPY_LINK_LIBRARIES endoscopy_tool_tracking LINK_LIBRARIES)
  target_link_libraries(endoscopy_tool_tracking_flowbenchmark PRIVATE ${ENDOSCOPY_LINK_LIBRARIES})
  target_include_directories(endoscopy_tool_tracking_flowbenchmark
                             PRIVATE "${CMAKE_SOURCE_DIR}/benchmarks/holoscan_flow_benchmarking")

  # Check if the machine is x86_64 or aarch64. IGX is considered to be aaarch64
  if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(INSTANCES "5") # run the benchmarking test with up to 5 instances of the application
  else()
    set(INSTANCES "3") # run the benchmarking test with up to 3 instances of the application
  endif()

  # Add the test
  add_test(NAME endoscopy_tool_tracking_cpp_benchmark_test
           COMMAND bash -c "for i in {1..${INSTANCES}}; do \
            python3 ${CMAKE_SOURCE_DIR}/benchmarks/holoscan_flow_benchmarking/benchmark.py \
              --run-command \
              \"applications/endoscopy_tool_tracking/cpp/endoscopy_tool_tracking_flowbenchmark --data ${HOLOHUB_DATA_DIR}/endoscopy\" \
              -i $i -r 5 -m 1000 -u --sched greedy \
              -d ${FLOW_BENCHMARK_DIR}/endoscopy_tool_tracking_$i;\
           done; \
           if [ ${INSTANCES} -eq 5 ]; then \
              python3 ${CMAKE_SOURCE_DIR}/benchmarks/holoscan_flow_benchmarking/analyze.py \
                --cdash --save-csv --max --avg --stddev --median --min --tail --flatness -p 90 95 99 99.9\
                -g ${FLOW_BENCHMARK_DIR}/endoscopy_tool_tracking_1/logger_greedy_* EndoscopyToolTracking1 \
                -g ${FLOW_BENCHMARK_DIR}/endoscopy_tool_tracking_2/logger_greedy_* EndoscopyToolTracking2 \
                -g ${FLOW_BENCHMARK_DIR}/endoscopy_tool_tracking_3/logger_greedy_* EndoscopyToolTracking3 \
                -g ${FLOW_BENCHMARK_DIR}/endoscopy_tool_tracking_4/logger_greedy_* EndoscopyToolTracking4 \
                -g ${FLOW_BENCHMARK_DIR}/endoscopy_tool_tracking_5/logger_greedy_* EndoscopyToolTracking5;\
          else \
              python3 ${CMAKE_SOURCE_DIR}/benchmarks/holoscan_flow_benchmarking/analyze.py \
              --cdash --save-csv --max --avg --stddev --median --min --tail --flatness -p 90 95 99 99.9\
                -g ${FLOW_BENCHMARK_DIR}/endoscopy_tool_tracking_1/logger_greedy_* EndoscopyToolTracking1 \
                -g ${FLOW_BENCHMARK_DIR}/endoscopy_tool_tracking_2/logger_greedy_* EndoscopyToolTracking2 \
                -g ${FLOW_BENCHMARK_DIR}/endoscopy_tool_tracking_3/logger_greedy_* EndoscopyToolTracking3;\
          fi; \
          for f in *.csv; do \
          python3 ${CMAKE_SOURCE_DIR}/benchmarks/holoscan_flow_benchmarking/generate_bar_graph.py $f;\
          done
            "
           WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
  # Set test timeout to 25 minutes
  set_tests_properties(endoscopy_tool_tracking_cpp_benchmark_test PROPERTIES TIMEOUT 1500)
endif()

# Install application and dependencies into the install/ directory for packaging
install(
  TARGETS endoscopy_tool_tracking
  DESTINATION bin/endoscopy_tool_tracking/cpp
)

install(
  FILES endoscopy_tool_tracking.yaml
  DESTINATION bin/endoscopy_tool_tracking/cpp
)

install(
  FILES endoscopy_tool_tracking_aja_overlay.yaml
  DESTINATION bin/endoscopy_tool_tracking/cpp
)
