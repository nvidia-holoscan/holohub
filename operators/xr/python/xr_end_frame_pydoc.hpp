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

#pragma once

#include <fmt/format.h>

namespace holoscan::doc {

namespace XrEndFrameOp {

// PyXrEndFrameOp Constructor
inline constexpr const char* doc_XrEndFrameOp_python = R"doc(
Operator representing an xr session end frame call

**==Named Inputs==**

xr_frame_state : holoscan::XrSession
XrSession obj representing the current state of the xr system

Parameters
----------
fragment : Fragment
The fragment that the operator belongs to.
xr_session : ``holoscan.XrSession``
The shared XrSession object
xr_composition_layers : ``IOSpec*``
composition layers
name : str, optional
The name of the operator.
)doc";

}  // namespace XrEndFrameOp
}  // namespace holoscan::doc
