# SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Shared headers used by the V4L2 depth operators. This is an implementation
# support target, not a discoverable operator project.
add_subdirectory(v4l2_depth_common)

add_holohub_operator(bgra_to_planar_tensor)
add_holohub_operator(cuda_compositor)
add_holohub_operator(depth_colorizer)
add_holohub_operator(frame_skipper)
add_holohub_operator(tensorrt_inference)
add_holohub_operator(v4l2_source)
add_holohub_operator(x11_display)
add_holohub_operator(yuyv_to_bgra)
