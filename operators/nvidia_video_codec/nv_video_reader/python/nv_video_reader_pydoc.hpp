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

namespace holoscan::doc::NvVideoReaderOp {

PYDOC(NvVideoReaderOp, R"doc(
Nv Video Reader operator.

This operator reads H.264/H.265 video files and emits raw encoded frames one at a time.
The frames remain in their compressed format for processing by downstream operators
like nv_video_decoder.
)doc")

// PyNvVideoReaderOp Constructor
PYDOC(NvVideoReaderOp_python, R"doc(
Nv Video Reader operator.

This operator reads H.264/H.265 video files and emits raw encoded frames one at a time.
The frames remain in their compressed format for processing by downstream operators
like nv_video_decoder.

Parameters
----------
directory : str
    Directory containing the video file to read (H.264/H.265 format).
filename : str
    Filename of the video file to read (H.264/H.265 format).
allocator : holoscan.core.Allocator
    Allocator for output buffers.
loop : bool, optional
    Loop the video file when end is reached. Default is False.
verbose : bool, optional
    Print detailed reader information. Default is False.
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

}  // namespace holoscan::doc::NvVideoReaderOp
