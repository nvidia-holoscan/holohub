/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <macros.hpp>

namespace holoscan::doc::OrsiSegmentationPreprocessorOp {

PYDOC(OrsiSegmentationPreprocessorOp, R"doc(
Operator carrying out pre-processing operations on segmentation outputs.
)doc")

// PySegmentationPreprocessorOp Constructor
PYDOC(OrsiSegmentationPreprocessorOp_python, R"doc(
Operator carrying out pre-processing operations on segmentation outputs.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
allocator : holoscan.resources.Allocator
    Memory allocator to use for the output.
in_tensor_name : str, optional
    Name of the input tensor.
network_output_type : str, optional
    Network output type (e.g. 'softmax').
data_format : str, optional
    Data format of network output.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    CudaStreamPool instance to allocate CUDA streams.
normalize_means : sequence of int
    Sequence of integers describing the color channel means to use when normalizing the tensor.
normalize_stds : sequence of int
    Sequence of integers describing the color channel stds to use when normalizing the tensor.
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

}  // namespace holoscan::doc::OrsiSegmentationPreprocessorOp

