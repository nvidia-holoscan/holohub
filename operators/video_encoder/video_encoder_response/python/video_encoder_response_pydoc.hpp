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

#ifndef PYHOLOHUB_OPERATORS_VIDEO_ENCODER_RESPONSE_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_VIDEO_ENCODER_RESPONSE_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace VideoEncoderResponseOp {

PYDOC(VideoEncoderResponseOp, R"doc(
Operator class to perform inference using an LSTM model.
)doc")

// PyVideoEncoderResponseOp Constructor
PYDOC(VideoEncoderResponseOp_python, R"doc(
Operator class to perform inference using an LSTM model.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
output_transmitter : holoscan.IOSpec
    Encoder I/O related Parameters.
pool : holoscan.Allocator
    Allocator for the output buffer.
videoencoder_context : holoscan.ops.VideoEncoderContext
    Encoder context.
outbuf_storage_type : int
    Output buffer storage type.
name : str, optional
    The name of the operator.
)doc")

PYDOC(gxf_typename, R"doc(
The GXF type name of the resource.

Returns
-------
str
    The GXF type name of the resource
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
spec : ``holoscan.core.OperatorSpec``
    The operator specification.
)doc")

}  // namespace VideoEncoderResponseOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_VIDEO_DECODER_RESPONSE_PYDOC_HPP
