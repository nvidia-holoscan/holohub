/* SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES.
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

#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

/**
 * @brief Pass-through operator for rendered color buffers.
 *
 * This operator exists to control the input port queue policy (size=1, pop),
 * preventing backpressure/deadlock when downstream visualization runs slower
 * than the producer.
 */
class ColorBufferPassthroughOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(ColorBufferPassthroughOp)

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;
};

}  // namespace holoscan::ops


