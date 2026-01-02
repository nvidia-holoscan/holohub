/* SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include "color_buffer_passthrough.hpp"

namespace holoscan::ops {

void ColorBufferPassthroughOp::setup(OperatorSpec& spec) {
  // This drops stale frames if queue is full
  spec.input<holoscan::gxf::Entity>("color_buffer_in",
                                   holoscan::IOSpec::kSizeOne,
                                   holoscan::IOSpec::QueuePolicy::kPop);

  spec.output<holoscan::gxf::Entity>("color_buffer_out");
}

void ColorBufferPassthroughOp::compute(InputContext& input, OutputContext& output,
                                       ExecutionContext& context) {
  auto color_message = input.receive<holoscan::gxf::Entity>("color_buffer_in");
  if (!color_message) { throw std::runtime_error("Failed to receive color buffer message"); }
  
  auto cuda_streams = input.receive_cuda_streams("color_buffer_in");

  output.emit(color_message.value(), "color_buffer_out");
}

}  // namespace holoscan::ops




