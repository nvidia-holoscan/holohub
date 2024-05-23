/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOHUB_OPERATORS_VIDEO_ENCODER_REQUEST_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_VIDEO_ENCODER_REQUEST_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace VideoEncoderRequestOp {

// PyVideoEncoderRequestOp Constructor
PYDOC(VideoEncoderRequestOp, R"doc(
Operator class to perform inference using an LSTM model.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
input_frame: holoscan.core.IOSpec
    Encoder I/O related Parameters.
inbuf_storage_type: int
    Input buffer storage type.
videoencoder_context: holoscan.ops.VideoEncoderContext
    Encoder context.
codec: int
    Video codec.
input_height: int
    Input height.
input_width: int
    Input width.
input_format: holoscan.ops.EncoderInputFormat
    Input format.
profile: int
    Video encoder profile.
bitrate: int
    Bitrate
framerate: int
    Framerate.
qp: int
    QP.
hw_preset_type: int
    Hardware preset type
level: int
    Level.
iframe_interval: int
    I-Frame interval.
rate_control_mode: int
    Rate control mode.
config: holoscan.ops.EncoderConfig
    Encoder configuration.
name : str, optional
    The name of the operator.
)doc")

}  // namespace VideoEncoderRequestOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_VIDEO_DECODER_REQUEST_PYDOC_HPP
