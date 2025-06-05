/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace holoscan::doc::NvVideoEncoderOp {

PYDOC(NvVideoEncoderOp, R"doc(
Nv Video Encoder operator.
)doc")

// PyNvVideoEncoderOp Constructor
PYDOC(NvVideoEncoderOp_python, R"doc(
Nv Video Encoder operator.

Parameters
----------
cuda_device_ordinal: int
    CUDA device ordinal.
allocator: holoscan.core.Allocator
    Allocator for output buffers.
width: int
    Width of the video frame.
height: int
    Height of the video frame.
codec: str
    Codec to use for encoding.
preset: str
    Preset to use for encoding.
bitrate: int
    Bitrate for encoding.
frame_rate: int
    Frame rate for encoding.
rate_control_mode: int
    Rate control mode for encoding.
multi_pass_encoding: int
    Multi-pass encoding for encoding.
name : str, optional
    The name of the operator.
)doc")

PYDOC(initialize, R"doc(
Initialize the operator.

This method is called only once when the operator is created for the first time,
and uses a light-weight initialization.
)doc")

PYDOC(setup, R"doc(
Define the operator specification.

Parameters
----------
spec : holoscan.core.OperatorSpec
    The operator specification.
)doc")

}  // namespace holoscan::doc::NvVideoEncoderOp
