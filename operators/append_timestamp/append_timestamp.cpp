/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <utility>

// If GXF has gxf/std/dlpack_utils.hpp it has DLPack support
#if __has_include("gxf/std/dlpack_utils.hpp")
#define GXF_HAS_DLPACK_SUPPORT 1
#include "gxf/std/tensor.hpp"
#else
#define GXF_HAS_DLPACK_SUPPORT 0
#include "holoscan/core/gxf/gxf_tensor.hpp"
#endif

#include <gxf/std/tensor.hpp>
#include "gxf/std/timestamp.hpp"

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "append_timestamp.hpp"

namespace holoscan::ops {

void AppendTimestampOp::setup(OperatorSpec& spec) {
  auto& input = spec.input<gxf::Entity>("in_tensor");
  auto& output = spec.output<gxf::Entity>("out_tensor");
}

void AppendTimestampOp::compute(InputContext& op_input, OutputContext& op_output,
                                ExecutionContext& context) {
  // Process input message
  // The type of `in_message` is 'holoscan::gxf::Entity'.
  auto in_message = op_input.receive<gxf::Entity>("in_tensor").value();

  // Add timestamp to the tensor
  auto timestamp =
      static_cast<nvidia::gxf::Entity&>(in_message).add<nvidia::gxf::Timestamp>("timestamp");
  if (timestamp) {
    (*timestamp)->pubtime = std::chrono::system_clock::now().time_since_epoch().count();
    (*timestamp)->acqtime = std::chrono::system_clock::now().time_since_epoch().count();
  }

  // Transmit the gxf video buffer to target
  op_output.emit(in_message, "out_tensor");
}

}  // namespace holoscan::ops
