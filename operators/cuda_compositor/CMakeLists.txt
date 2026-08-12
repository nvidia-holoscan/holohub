# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(cuda_compositor cuda_compositor.cu)
add_library(holoscan::cuda_compositor ALIAS cuda_compositor)

target_compile_features(cuda_compositor PUBLIC cxx_std_20)
set_target_properties(cuda_compositor PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_compile_options(cuda_compositor
    PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
)
target_include_directories(cuda_compositor
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(cuda_compositor
    PUBLIC holoscan::v4l2_depth_common
    PRIVATE CUDA::cudart
)
