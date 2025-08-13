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

namespace holoscan::doc::TensorToFileOp {

PYDOC(TensorToFileOp, R"doc(
Nv Video Writer operator.

This operator writes H.264/H.265 elementary stream files from encoded video frames.
Takes encoded frame tensors as input and writes them directly to elementary stream files
that can be played with standard video players.
)doc")

// PyTensorToFileOp Constructor
PYDOC(TensorToFileOp_python, R"doc(
Nv Video Writer operator.

This operator writes H.264/H.265 elementary stream files from encoded video frames.
Takes encoded frame tensors as input (typically from NvVideoEncoderOp) and writes
them directly to elementary stream files that can be played with standard video players.

Parameters
----------
tensor_name : str
    Name of the tensor to write to the file.
output_file : str
    Output file path for the elementary stream (e.g., "output.h264" or "output.h265").
allocator : holoscan.core.Allocator
    Allocator for output buffers.
verbose : bool, optional
    Print detailed writer information including frame count and bytes written. Default is False.
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

}  // namespace holoscan::doc::TensorToFileOp
