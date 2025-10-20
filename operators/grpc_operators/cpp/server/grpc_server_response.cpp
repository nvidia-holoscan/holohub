/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "grpc_server_response.hpp"

namespace holoscan::ops {

void GrpcServerResponseOp::setup(OperatorSpec& spec) {
  spec.input<nvidia::gxf::Entity>("input", IOSpec::kAnySize);

  spec.param(response_queue_, "response_queue", "Response Queue", "Outgoing gRPC results.");

  cuda_stream_handler_.define_params(spec);
}

void GrpcServerResponseOp::compute(InputContext& op_input, OutputContext& op_output,
                                   ExecutionContext& context) {
  auto tensors = 0;
  auto response = std::make_shared<EntityResponse>();

  auto input_messages = op_input.receive<std::vector<holoscan::gxf::Entity>>("input").value();

  for (auto&& message : input_messages) {
    holoscan::ops::TensorProto::tensor_to_entity_response(
        message, response, cuda_stream_handler_.get_cuda_stream(fragment()->executor().context()));
    tensors++;
  }

  if (tensors > 0) {
    HOLOSCAN_LOG_DEBUG("Sending response with {} tensors.", tensors);
    response_queue_->push(response);
  }
}

}  // namespace holoscan::ops
