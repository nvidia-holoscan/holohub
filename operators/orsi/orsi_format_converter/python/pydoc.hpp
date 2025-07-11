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

#include "macros.hpp"

namespace holoscan::doc::OrsiFormatConverterOp {

PYDOC(OrsiFormatConverterOp, R"doc(
Format conversion operator.
)doc")

// PyOrsiFormatConverterOp Constructor
PYDOC(OrsiFormatConverterOp_python, R"doc(
Format conversion operator.

Parameters
----------
fragment : holoscan.core.Fragment
    The fragment that the operator belongs to.
pool : holoscan.resources.Allocator
    Memory pool allocator used by the operator.
out_dtype : str
    Destination data type (e.g. "RGB888" or "RGBA8888").
in_dtype : str, optional
    Source data type (e.g. "RGB888" or "RGBA8888").
in_tensor_name : str, optional
    The name of the input tensor.
out_tensor_name : str, optional
    The name of the output tensor.
scale_min : float, optional
    Output will be clipped to this minimum value.
scale_max : float, optional
    Output will be clipped to this maximum value.
alpha_value : int, optional
    Unsigned integer in range [0, 255], indicating the alpha channel value to use
    when converting from RGB to RGBA.
resize_height : int, optional
    Desired height for the (resized) output. Height will be unchanged if `resize_height` is 0.
resize_width : int, optional
    Desired width for the (resized) output. Width will be unchanged if `resize_width` is 0.
resize_mode : int, optional
    Resize mode enum value corresponding to NPP's nppiInterpolationMode (default=NPPI_INTER_CUBIC).
channel_order : sequence of int
    Sequence of integers describing how channel values are permuted.
src_roi_rec : sequence of int
    Sequence of integers describing region of interest to crop.
output_img_size : sequence of int
    Sequence of integers describing size of output image after resizing.
cuda_stream_pool : holoscan.resources.CudaStreamPool, optional
    CudaStreamPool instance to allocate CUDA streams.
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

}  // namespace holoscan::doc::OrsiFormatConverterOp
