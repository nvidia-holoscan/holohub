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

#ifndef PYHOLOHUB_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace ToolTrackingPostprocessorOp {

// PyToolTrackingPostprocessorOp Constructor
PYDOC(ToolTrackingPostprocessorOp_python, R"doc(
Operator performing post-processing for the endoscopy tool tracking demo.

**==Named Inputs==**

    in : nvidia::gxf::Entity containing multiple nvidia::gxf::Tensor
        Must contain input tensors named "probs", "scaled_coords" and "binary_masks" that
        correspond to the output of the LSTMTensorRTInfereceOp as used in the endoscopy
        tool tracking example applications.

**==Named Outputs==**

    out : nvidia::gxf::Tensor
        Binary mask and coordinates tensor, stored on the device (GPU).

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
device_allocator : ``holoscan.resources.Allocator``
    Output allocator used on the device side.
min_prob : float, optional
    Minimum probability (in range [0, 1]). Default value is 0.5.
overlay_img_colors : sequence of sequence of float, optional
    Color of the image overlays, a list of RGB values with components between 0 and 1.
    The default value is a qualitative colormap with a sequence of 12 colors.
cuda_stream_pool : ``holoscan.resources.CudaStreamPool``, optional
    `holoscan.resources.CudaStreamPool` instance to allocate CUDA streams.
    Default value is ``None``.
name : str, optional
    The name of the operator.
)doc")
}  // namespace ToolTrackingPostprocessorOp


}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_TOOL_TRACKING_POSTPROCESSOR_PYDOC_HPP
