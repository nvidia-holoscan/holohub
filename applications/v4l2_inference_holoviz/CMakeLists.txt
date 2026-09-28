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
project(v4l2_inference_holoviz LANGUAGES CXX)

include(CTest)
option(V4L2_INFERENCE_HOLOVIZ_BUILD_APP "Build the Holoscan 5.x camera application" ON)

if(V4L2_INFERENCE_HOLOVIZ_BUILD_APP)
  find_package(holoscan 5.0 REQUIRED CONFIG COMPONENTS sensor_io inference visualization)
  find_package(holoscan-camera 0.3 REQUIRED CONFIG)
  if(NOT TARGET holoscan::op_tensor_preprocessor)
    message(FATAL_ERROR
      "This app requires the inference module's TensorPreprocessorOp with ImageT support. "
      "Install main-5x with the merged TensorPreprocessorOp; see this app's README.")
  endif()

  add_executable(v4l2_inference_holoviz src/main.cpp)
  target_compile_features(v4l2_inference_holoviz PRIVATE cxx_std_20)
  target_link_libraries(v4l2_inference_holoviz PRIVATE
    holoscan::core
    holoscan::sensor_io
    holoscan::v4l2_capture_op
    holoscan::op_tensor_preprocessor
    holoscan::op_inference
    holoscan::visualization::op_holoviz)
  install(TARGETS v4l2_inference_holoviz RUNTIME DESTINATION bin)

  if(BUILD_TESTING)
    add_test(NAME v4l2_inference_holoviz_help COMMAND v4l2_inference_holoviz --help)
    set_tests_properties(v4l2_inference_holoviz_help PROPERTIES
      PASS_REGULAR_EXPRESSION "V4L2 YUYV camera" TIMEOUT 10)
    add_test(NAME v4l2_inference_holoviz_graph COMMAND v4l2_inference_holoviz --validate)
    set_tests_properties(v4l2_inference_holoviz_graph PROPERTIES
      PASS_REGULAR_EXPRESSION "graph validation complete" LABELS "gpu" TIMEOUT 30)
  endif()
endif()

if(BUILD_TESTING)
  add_executable(v4l2_inference_holoviz_unit tests/test_app.cpp)
  target_include_directories(v4l2_inference_holoviz_unit PRIVATE src)
  target_compile_features(v4l2_inference_holoviz_unit PRIVATE cxx_std_20)
  add_test(NAME v4l2_inference_holoviz_unit COMMAND v4l2_inference_holoviz_unit)
  set_tests_properties(v4l2_inference_holoviz_unit PROPERTIES TIMEOUT 10)

  set(V4L2_INFERENCE_HOLOVIZ_TEST_MODEL
    "${CMAKE_CURRENT_SOURCE_DIR}/../../data/v4l2_inference_holoviz/identity_model.onnx"
    CACHE FILEPATH "Inference module example identity model to validate")
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    execute_process(COMMAND "${Python3_EXECUTABLE}" -c "import onnx.reference"
      RESULT_VARIABLE _onnx_available OUTPUT_QUIET ERROR_QUIET)
    if(_onnx_available EQUAL 0 AND EXISTS "${V4L2_INFERENCE_HOLOVIZ_TEST_MODEL}")
      add_test(NAME v4l2_inference_holoviz_identity_model
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_identity_model.py"
          --model "${V4L2_INFERENCE_HOLOVIZ_TEST_MODEL}")
      set_tests_properties(v4l2_inference_holoviz_identity_model PROPERTIES TIMEOUT 30)
    else()
      message(STATUS "Skipping identity model test: Python onnx or the SDK model is unavailable")
    endif()
  endif()
endif()
