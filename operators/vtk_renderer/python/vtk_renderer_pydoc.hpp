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

#ifndef PYHOLOHUB_OPERATORS_VTK_RENDERER_PYDOC_HPP
#define PYHOLOHUB_OPERATORS_VTK_RENDERER_PYDOC_HPP

#include <string>

#include "macros.hpp"

namespace holoscan::doc {

namespace VtkRendererOp {

PYDOC(VtkRendererOp, R"doc(
Operator for using VTK for rendering.

**==Named Inputs==**

    annotations: Input channel for the annotations, type `gxf::Tensor`
    videostream: Input channel for the videostream, type `gxf::Tensor`

Parameters
----------
labels: std::vector<std::string>>
    labels to be displayed on the rendered image.
width: int
    width of the renderer window.
eight: int
    height of the renderer window.
window_name: std::string
    Compositor window name.
name : str, optional
    The name of the operator.
)doc")

}  // namespace VtkRendererOp

}  // namespace holoscan::doc

#endif  // PYHOLOHUB_OPERATORS_VTK_RENDERER_PYDOC_HPP
