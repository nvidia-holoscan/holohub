# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

add_library(bgra_to_planar_tensor bgra_to_planar_tensor.cu)
add_library(holoscan::bgra_to_planar_tensor ALIAS bgra_to_planar_tensor)

target_compile_features(bgra_to_planar_tensor PUBLIC cxx_std_20)
set_target_properties(bgra_to_planar_tensor PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(bgra_to_planar_tensor
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
)
target_link_libraries(bgra_to_planar_tensor
    PUBLIC holoscan::v4l2_depth_common
    PRIVATE CUDA::cudart
)
