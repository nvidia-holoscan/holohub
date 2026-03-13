/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#ifndef HOLOHUB_IMX274_GPU_RESIDENT_NOOP_SINK_OP
#define HOLOHUB_IMX274_GPU_RESIDENT_NOOP_SINK_OP

#include <holoscan/holoscan.hpp>

namespace imx274_gpu_resident {

class NoopSinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NoopSinkOp);

  NoopSinkOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<holoscan::gxf::Entity>("input"); }

  void compute(holoscan::InputContext& input, holoscan::OutputContext&,
               holoscan::ExecutionContext&) override {
    (void)input.receive<holoscan::gxf::Entity>("input");
  }
};

}  // namespace imx274_gpu_resident

#endif /* HOLOHUB_IMX274_GPU_RESIDENT_NOOP_SINK_OP */
