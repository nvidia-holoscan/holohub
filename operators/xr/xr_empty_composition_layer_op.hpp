/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef XR_EMPTY_COMPOSITION_LAYER_OP_HPP
#define XR_EMPTY_COMPOSITION_LAYER_OP_HPP

#include <memory>

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

// Generates an empty composition layer for OpenXR frame completion.
// This operator is required to complete the OpenXR frame lifecycle when
// no visual rendering is being performed. It outputs a nullptr composition layer.
class XrEmptyCompositionLayerOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(XrEmptyCompositionLayerOp)

  XrEmptyCompositionLayerOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
};

}  // namespace holoscan::ops

#endif  // XR_EMPTY_COMPOSITION_LAYER_OP_HPP

