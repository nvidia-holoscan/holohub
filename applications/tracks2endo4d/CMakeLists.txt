# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(tracks2endo4d LANGUAGES CXX CUDA)

find_package(holoscan 3.0 REQUIRED CONFIG
               PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

#Visualization
include(pybind11_add_holohub_module)

# Build tracks2endo4d_viz library (CUDA for thrust support)
add_library(tracks2endo4d_viz SHARED viz_cpp/viz.cu)
target_link_libraries(tracks2endo4d_viz
  PRIVATE
    holoscan::core
    holoscan::viz
    holoscan::viz::imgui
)

# Build tracks2endo4d_viz module
pybind11_add_holohub_module(
  CPP_CMAKE_TARGET tracks2endo4d_viz
  CLASS_NAME "VizOp"
  SOURCES viz_cpp/viz_bind.cpp
)

# Default to convert engine
option(CONVERT_ENGINE "Convert engine" ON)

# Download the sample data
if(HOLOHUB_DOWNLOAD_DATASETS)
  message(STATUS "Downloading sample data to ${HOLOHUB_DATA_DIR}")
  include(holoscan_download_data)
  holoscan_download_data(tracks2endo4d
    URL nvstaging/holoscan/holoscan_tracking_sample_data:1.2
    DOWNLOAD_NAME holoscan_tracking_sample_data_1.2.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )
endif()

if(CONVERT_ENGINE)
  message(STATUS "Converting engine to TensorRT")
  add_custom_target(convert_engine_tapnext ALL
    WORKING_DIRECTORY ${HOLOHUB_DATA_DIR}/tracks2endo4d/models
    COMMAND cp /models/* ${HOLOHUB_DATA_DIR}/tracks2endo4d/models/
    COMMAND trtexec --onnx=tapnext_forward.onnx --saveEngine=tapnext_forward.bf16.engine --bf16 --skipInference
    COMMAND trtexec --onnx=tapnext_init.onnx --saveEngine=tapnext_init.bf16.engine --bf16 --skipInference
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/tracks2endo4d/models/tapnext_forward.bf16.engine"
                "${HOLOHUB_DATA_DIR}/tracks2endo4d/models/tapnext_init.bf16.engine"
  )
  add_custom_target(convert_engine_tracks_to_4d ALL
    WORKING_DIRECTORY ${HOLOHUB_DATA_DIR}/tracks2endo4d/models
    COMMAND trtexec --onnx=tracksto4d.onnx --saveEngine=tracksto4d.bf16.engine --bf16 --skipInference
    BYPRODUCTS "${HOLOHUB_DATA_DIR}/tracks2endo4d/models/tracksto4d.bf16.engine"
  )

  # Ensure data is downloaded before converting
  if(HOLOHUB_DOWNLOAD_DATASETS)
    add_dependencies(convert_engine_tapnext tracks2endo4d_data)
    add_dependencies(convert_engine_tracks_to_4d tracks2endo4d_data)
  endif()
endif()

# --------------------------------------------------------------------------
# Tracks2Endo4D Application
# --------------------------------------------------------------------------

# Copy application file
add_custom_target(python_tracks2endo4d_app ALL
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/tracks2endo4d_app.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "tracks2endo4d_app.py"
  BYPRODUCTS "tracks2endo4d_app.py"
)

# Copy dependencies
add_custom_target(python_tracks2endo4d_app_deps
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/config.yaml" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/postprocessing.py" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/preprocessing.py" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/tracking.py" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/tracks_assembler.py" ${CMAKE_CURRENT_BINARY_DIR}
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/tracks_3d.py" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "config.yaml" "postprocessing.py" "tracking.py" "preprocessing.py"
          "tracks_assembler.py" "tracks_3d.py"
  BYPRODUCTS "config.yaml" "postprocessing.py" "tracking.py" "preprocessing.py"
   "tracks_assembler.py" "tracks_3d.py"
)

add_dependencies(python_tracks2endo4d_app python_tracks2endo4d_app_deps)

if(CONVERT_ENGINE)
  add_dependencies(python_tracks2endo4d_app convert_engine_tapnext)
  add_dependencies(python_tracks2endo4d_app convert_engine_tracks_to_4d)
endif()

# Install
install(
  FILES tracks2endo4d_app.py
        config.yaml
        postprocessing.py
        preprocessing.py
        tracking.py
        tracks_3d.py
        tracks_assembler.py
  DESTINATION bin/tracks2endo4d
)

# --------------------------------------------------------------------------
# Testing
# --------------------------------------------------------------------------

if(BUILD_TESTING)
  # Run the application for a limited number of frames and verify the output
  add_test(NAME tracks2endo4d_python_test
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/tracks2endo4d_app.py
            --data ${HOLOHUB_DATA_DIR}/tracks2endo4d
            --model ${HOLOHUB_DATA_DIR}/tracks2endo4d
            --count 30
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  )
  set_property(TEST tracks2endo4d_python_test PROPERTY ENVIRONMENT
    "PYTHONPATH=${GXF_LIB_DIR}/../python/lib:${CMAKE_BINARY_DIR}/python/lib:${CMAKE_SOURCE_DIR}")

  set_tests_properties(tracks2endo4d_python_test PROPERTIES
    PASS_REGULAR_EXPRESSION "Reach end of file or playback count reaches to the limit. Stop ticking."
    FAIL_REGULAR_EXPRESSION "[^a-z]Error;ERROR;Failed"
  )
endif()
