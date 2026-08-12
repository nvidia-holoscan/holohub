# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(depth_colorizer depth_colorizer.cu)
add_library(holoscan::depth_colorizer ALIAS depth_colorizer)

target_compile_features(depth_colorizer PUBLIC cxx_std_20)
set_target_properties(depth_colorizer PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(depth_colorizer
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(depth_colorizer
    PUBLIC holoscan::v4l2_depth_common
    PRIVATE CUDA::cudart
)
