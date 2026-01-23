/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "xr_empty_composition_layer_op.hpp"

#include <memory>

#include "openxr/openxr.hpp"

namespace holoscan::ops {

void XrEmptyCompositionLayerOp::setup(OperatorSpec& spec) {
  spec.output<std::shared_ptr<xr::CompositionLayerBaseHeader>>("xr_composition_layer");
}

void XrEmptyCompositionLayerOp::compute(InputContext& input, OutputContext& output,
                                         ExecutionContext& context) {
  // Emit nullptr to represent an empty composition layer.
  // This allows the XrEndFrameOp to handle frames with no visual content.
  std::shared_ptr<xr::CompositionLayerBaseHeader> empty_layer = nullptr;
  output.emit(empty_layer, "xr_composition_layer");
}

}  // namespace holoscan::ops

