/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "xr_end_frame_op.hpp"

#include <memory>
#include <vector>

#include "openxr/openxr.hpp"

namespace holoscan::ops {

void XrEndFrameOp::setup(OperatorSpec& spec) {
  spec.input<xr::FrameState>("xr_frame_state");
  spec.input<std::vector<std::shared_ptr<xr::CompositionLayerBaseHeader>>>("xr_composition_layers",
                                                                           IOSpec::kAnySize);

  spec.param(xr_session_, "xr_session", "OpenXR Session", "OpenXR Session");
}

void XrEndFrameOp::compute(InputContext& input, OutputContext& output, ExecutionContext& context) {
  std::shared_ptr<holoscan::XrSession> xr_session = xr_session_.get();

  auto xr_frame_state = input.receive<xr::FrameState>("xr_frame_state");
  auto xr_composition_layers =
      input.receive<std::vector<std::shared_ptr<xr::CompositionLayerBaseHeader>>>(
          "xr_composition_layers");

  // Submit composition layers to the XR device for display.
  std::vector<xr::CompositionLayerBaseHeader*> layers;
  if (xr_composition_layers.has_value()) {
    for (std::shared_ptr<xr::CompositionLayerBaseHeader>& layer : xr_composition_layers.value()) {
      if (layer != nullptr) {
        layers.push_back(layer.get());
      }
    }
  }
  xr_session->get().endFrame({
      xr_frame_state->predictedDisplayTime,
      xr_session->blend_mode(),
      static_cast<uint32_t>(layers.size()),
      layers.data(),
  });
}

}  // namespace holoscan::ops
