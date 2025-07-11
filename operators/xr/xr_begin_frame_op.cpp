/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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

#include "xr_begin_frame_op.hpp"

#include <memory>

#include "openxr/openxr.hpp"

namespace holoscan::ops {

void XrBeginFrameOp::setup(OperatorSpec& spec) {
  spec.output<xr::FrameState>("xr_frame_state");

  spec.param(xr_session_, "xr_session", "OpenXR Session", "OpenXR Session");
}

void XrBeginFrameOp::compute(InputContext& input, OutputContext& output,
                             ExecutionContext& context) {
  std::shared_ptr<holoscan::XrSession> xr_session = xr_session_.get();

  // Synchronize the application with the XR device.
  xr::FrameState xr_frame_state = xr_session->get().waitFrame({});

  // Signal that the frame has begun rendering.
  xr_session->get().beginFrame({});

  // Emit the frame state to be consumed by XrEndFrameOp.
  output.emit(xr_frame_state, "xr_frame_state");
}

}  // namespace holoscan::ops
