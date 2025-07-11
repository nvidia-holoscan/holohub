/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef PYHOLOHUB_OPERATORS_TENSOR_TO_VIDEO_BUFFER_OP_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_TENSOR_TO_VIDEO_BUFFER_OP_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace TensorToVideoBufferOp {

PYDOC(TensorToVideoBufferOp, R"doc(
Operator class to convert Tensor to VideoBuffer.


**==Named Inputs==**

    in_tensor : gxf::Entity
    
**==Named Outputs==**

    out_video_buffer : gxf::Entity

Parameters
----------
fragment : Fragment
    The fragment that the operator belongs to.
in_tensor_name: str
    Input tensor name.
video_format_: str
    Video format.
name : str, optional
    The name of the operator.

)doc")

}  // namespace TensorToVideoBufferOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_VIDEO_DECODER_RESPONSE_PYDOC_HPP
