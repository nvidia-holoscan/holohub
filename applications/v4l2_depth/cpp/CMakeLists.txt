# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_executable(v4l2_depth v4l2_depth.cu)
target_compile_features(v4l2_depth PRIVATE cxx_std_20)
set_target_properties(v4l2_depth
    PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_EXTENSIONS OFF
)
target_compile_options(v4l2_depth
    PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda;--expt-relaxed-constexpr>
)
target_link_libraries(v4l2_depth
    PRIVATE
        holoscan::bgra_to_planar_tensor
        holoscan::cuda_compositor
        holoscan::depth_colorizer
        holoscan::frame_skipper
        holoscan::tensorrt_inference
        holoscan::v4l2_source
        holoscan::x11_display
        holoscan::yuyv_to_bgra
        holoscan::core
        Threads::Threads
)
