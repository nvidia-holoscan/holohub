# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_holohub_application(v4l2_depth
    DEPENDS
        OPERATORS
            bgra_to_planar_tensor
            cuda_compositor
            depth_colorizer
            frame_skipper
            tensorrt_inference
            v4l2_source
            x11_display
            yuyv_to_bgra
)
