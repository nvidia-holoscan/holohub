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

#ifndef XR_END_FRAME_OP_HPP
#define XR_END_FRAME_OP_HPP

#include <memory>

#include "holoscan/holoscan.hpp"
#include "xr_session.hpp"

namespace holoscan::ops {

// Ends an OpenXR frame by calling xrEndFrame on each tick.
class XrEndFrameOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(XrEndFrameOp)

  XrEndFrameOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<holoscan::XrSession>> xr_session_;
};

}  // namespace holoscan::ops

#endif  // XR_END_FRAME_OP_HPP
