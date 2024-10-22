/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOHUB_OPERATORS_VISUALIZER_ICARDIO_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_VISUALIZER_ICARDIO_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace VisualizerICardioOp {

PYDOC(VisualizerICardioOp, R"doc(
iCardio Multi-AI demo application visualization operator.
)doc")

// PyVisualizerICardioOp Constructor
PYDOC(VisualizerICardioOp_python, R"doc(
iCardio Multi-AI demo application visualization operator.

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
allocator : ``holoscan.resources.Allocator``
    Memory allocator to use for the output.
in_tensor_names : sequence of str, optional
    Names of input tensors in the order to be fed into the operator.
out_tensor_names : sequence of str, optional
    Names of output tensors in the order to be fed into the operator.
input_on_cuda : bool, optional
    Boolean indicating whether the input tensors are on the GPU.
data_dir: string
    Path to the data for the iCardio logo.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    CudaStreamPool instance to allocate CUDA streams.
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

}  // namespace VisualizerICardioOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_VISUALIZER_ICARDIO_PYDOC_HPP
