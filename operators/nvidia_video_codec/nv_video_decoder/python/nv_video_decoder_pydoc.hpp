/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <string>

#include "macros.hpp"

namespace holoscan::doc::NvVideoDecoderOp {

PYDOC(NvVideoDecoderOp, R"doc(
Decode compressed video using the NVIDIA Video Codec SDK.

When ``codec`` is set to ``"H264"`` or ``"HEVC"``, input tensors are fed
straight to the CUVID parser. ``packetized_input_mode`` describes whether those
inputs are arbitrary byte-stream chunks or complete encoded access units.
``packetized_low_latency`` independently controls CUVID's display/reordering
policy. Packetized input tensors must use host-accessible host or system
storage; device-backed encoded payloads are not supported. Packetized
bitstreams must decode to 8-bit 4:2:0 NV12.
)doc")

// PyNvVideoDecoderOp Constructor
PYDOC(NvVideoDecoderOp_python, R"doc(
Decode compressed video using the NVIDIA Video Codec SDK.

Parameters
----------
cuda_device_ordinal : int
    CUDA device ordinal.
allocator : holoscan.core.Allocator
    Allocator for output buffers.
verbose : bool, optional
    Print detailed decoder information. Default is False.
name : str, optional
    The name of the operator.
codec : str, optional
    Codec for direct packetized input. Set to ``"H264"`` or ``"HEVC"`` to
    bypass the FFmpeg demuxer and feed each input tensor directly to NVDEC.
    Packetized input tensors must use host-accessible host or system storage;
    device-backed encoded payloads are not supported. Leave empty for the
    existing demuxed/file streaming behavior. Only bitstreams decoded to 8-bit
    4:2:0 NV12 are supported by the packetized path.
packetized_input_mode : str, optional
    Framing of direct packetized input. ``"stream"`` (default) treats input
    tensors as arbitrary byte-stream chunks and lets CUVID determine picture
    boundaries. ``"access_unit"`` declares that every input tensor contains
    exactly one complete encoded access unit; the operator then submits the
    packet with ``CUVID_PKT_ENDOFPICTURE`` so the parser can complete that
    picture without waiting for the next input tensor.

    Only use ``"access_unit"`` when the producer guarantees one complete access
    unit per input tensor. Using it with fragmented input can create incorrect
    parser boundaries or decode failures.
packetized_low_latency : bool, optional
    Decoder display policy for direct packetized input. The default, ``False``,
    preserves normal CUVID display reordering and supports streams containing
    B-frames. Set to ``True`` only for low-latency bitstreams without B-frames,
    such as All-Intra or IPPP streams. This option is independent of
    ``packetized_input_mode`` and does not change end-of-picture signaling.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization. Invalid ``packetized_input_mode`` values
raise an exception during initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace holoscan::doc::NvVideoDecoderOp
