/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, XRlabs. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef PYHOLOHUB_OPERATORS_ST2110_SOURCE_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_ST2110_SOURCE_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace ST2110SourceOp {

PYDOC(ST2110SourceOp_python, R"doc(
Operator to receive SMPTE ST 2110-20 uncompressed video streams via Linux sockets.

This operator uses standard Linux UDP sockets with CUDA pinned memory to receive
ST 2110-20 video streams over IP networks. It handles multicast subscription,
RTP packet reception, frame reassembly, and outputs video frames for downstream
processing.

**==Named Outputs==**

    raw_output : holoscan::gxf::Entity with Tensor
        Raw video frame buffer with format metadata (always available).
        Format depends on stream_format parameter (default: YCbCr-4:2:2-10bit).

    rgba_output : nvidia::gxf::VideoBuffer (optional)
        RGBA 8-bit converted frame, enabled via ``enable_rgba_output`` parameter.
        Suitable for visualization with Holoviz.

    nv12_output : nvidia::gxf::VideoBuffer (optional)
        NV12 8-bit converted frame, enabled via ``enable_nv12_output`` parameter.
        Suitable for video encoding.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
multicast_address : str, optional
    Multicast IP address for ST 2110 stream (e.g., "239.255.66.60").
    Default value is ``"239.0.0.1"``.
port : int, optional
    UDP port for ST 2110 stream. Default value is ``5004``.
interface_name : str, optional
    Linux network interface name (e.g., "mgbe0_0", "eth0").
    Default value is ``"eth0"``.
width : int, optional
    Width of the video stream in pixels. Default value is ``1920``.
height : int, optional
    Height of the video stream in pixels. Default value is ``1080``.
framerate : int, optional
    Expected frame rate of the video stream. Default value is ``60``.
stream_format : str, optional
    Input stream format. Supported: "YCbCr-4:2:2-10bit", "YCbCr-4:2:2-8bit", "RGBA-8bit".
    Default value is ``"YCbCr-4:2:2-10bit"``.
enable_rgba_output : bool, optional
    Enable RGBA conversion and emission on rgba_output port.
    Default value is ``False``.
enable_nv12_output : bool, optional
    Enable NV12 conversion and emission on nv12_output port.
    Default value is ``False``.
batch_size : int, optional
    Number of packets to receive per compute() call. Default value is ``1000``.
max_packet_size : int, optional
    Maximum size of ST 2110 packets in bytes. Default value is ``1514``.
header_size : int, optional
    Size of L2-L4 headers (Ethernet + IP + UDP) in bytes. Default value is ``42``.
rtp_header_size : int, optional
    Size of RTP header in bytes. Default value is ``12``.
enable_reorder_kernel : bool, optional
    Enable CUDA kernel for packet reordering. Default value is ``True``.
name : str, optional
    The name of the operator. Default value is ``"st2110_source"``.

Notes
-----
This operator requires:
    - Network interface with multicast support
    - Properly configured multicast routing on the host
    - Sufficient socket buffer size (see net.core.rmem_max)

The operator is supported on both x86_64 and aarch64 (ARM64) platforms,
including NVIDIA Thor AGX with MGBE.
)doc")

}  // namespace ST2110SourceOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_ST2110_SOURCE_PYDOC_HPP
