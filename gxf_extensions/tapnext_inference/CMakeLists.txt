# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(gxf_tapnext_inference)

find_package(holoscan REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Create library
add_library(gxf_tapnext_inference_lib SHARED
  tapnext_inference.cpp
  tapnext_inference.hpp
)

set_target_properties(gxf_tapnext_inference_lib PROPERTIES CXX_STANDARD 17)

target_link_libraries(gxf_tapnext_inference_lib
  PUBLIC
    CUDA::cudart
    GXF::cuda
    GXF::std
    nvinfer
    nvinfer_plugin
    nvonnxparser
    yaml-cpp
    holoscan::core
)

# Create extension
add_library(gxf_tapnext_inference SHARED
  tapnext_extension.cpp
)
target_link_libraries(gxf_tapnext_inference
  PUBLIC gxf_tapnext_inference_lib
)

install(TARGETS gxf_tapnext_inference_lib
                gxf_tapnext_inference
        DESTINATION lib/gxf_extensions
)
