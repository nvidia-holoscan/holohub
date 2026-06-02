# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(gxf_lstm_tensor_rt_inference)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

find_library(TENSORRT_LIBRARY_INFER nvinfer nvinfer_10
    HINTS 
        $ENV{TENSORRT_ROOT}
        ${TENSORRT_ROOT}
        ${holoscan_DIR}/../..
    PATH_SUFFIXES lib lib64 lib/x64
)

find_library(TENSORRT_LIBRARY_INFER_PLUGIN nvinfer_plugin nvinfer_plugin_10
    HINTS 
        $ENV{TENSORRT_ROOT}
        ${TENSORRT_ROOT}
        ${holoscan_DIR}/../..
    PATH_SUFFIXES lib lib64 lib/x64
)

find_library(TENSORRT_LIBRARY_ONNXPARSER nvonnxparser nvonnxparser_10
    HINTS 
        $ENV{TENSORRT_ROOT}
        ${TENSORRT_ROOT}
        ${holoscan_DIR}/../..
    PATH_SUFFIXES lib lib64 lib/x64
)

# Create library
add_library(gxf_lstm_tensor_rt_inference_lib SHARED
  tensor_rt_inference.cpp
  tensor_rt_inference.hpp
)

set_target_properties(gxf_lstm_tensor_rt_inference_lib PROPERTIES 
    CXX_STANDARD 17
    WINDOWS_EXPORT_ALL_SYMBOLS ON
)

target_link_directories(gxf_lstm_tensor_rt_inference_lib PUBLIC "${holoscan_DIR}/../..")
#message(FATAL_ERROR "${holoscan_DIR}/../..")

target_link_libraries(gxf_lstm_tensor_rt_inference_lib
  PUBLIC
    CUDA::cudart
    GXF::cuda
    GXF::std
    ${TENSORRT_LIBRARY_INFER}
    ${TENSORRT_LIBRARY_INFER_PLUGIN}
    ${TENSORRT_LIBRARY_ONNXPARSER}
    holoscan::core  # included only as a way to find dlpack/dlpack.h for GXF::std
)

# Create extension
add_library(gxf_lstm_tensor_rt_inference SHARED
  lstm_tensor_rt_extension.cpp
)

# On Windows, use a .def file to export GxfExtensionFactory
if(WIN32)
  target_sources(gxf_lstm_tensor_rt_inference PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/gxf_lstm_tensor_rt_inference.def
  )
endif()

target_link_libraries(gxf_lstm_tensor_rt_inference
  PUBLIC gxf_lstm_tensor_rt_inference_lib
)

install(TARGETS gxf_lstm_tensor_rt_inference_lib
                gxf_lstm_tensor_rt_inference
        DESTINATION lib/gxf_extensions
                )
