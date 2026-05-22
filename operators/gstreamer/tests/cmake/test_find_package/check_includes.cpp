// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Trivial compile-only check: verifies that the installed include path(s) exported
// by holoscan::holoscan_gstreamer_bridge is correct.  If the INSTALL_INTERFACE
// path is wrong this translation unit will fail to compile.
#include <gst_src_resource.hpp>
#include <holoscan/operators/gstreamer/gst_video_recorder_op.hpp>

int main() {
    return 0;
}
