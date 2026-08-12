# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

find_package(CUDAToolkit REQUIRED)
find_package(TensorRT REQUIRED COMPONENTS nvinfer nvonnxparser)

add_library(tensorrt_inference tensorrt_inference.cpp)
add_library(holoscan::tensorrt_inference ALIAS tensorrt_inference)

target_compile_features(tensorrt_inference PUBLIC cxx_std_20)
target_include_directories(tensorrt_inference
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(tensorrt_inference
    PUBLIC holoscan::v4l2_depth_common
    PRIVATE
        CUDA::cudart
        TensorRT::nvinfer
        TensorRT::nvonnxparser
)
