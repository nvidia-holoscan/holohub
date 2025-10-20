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

#include "grpc_client_request.hpp"

namespace holoscan::ops {

void GrpcClientRequestOp::setup(OperatorSpec& spec) {
  spec.input<nvidia::gxf::Entity>("input");

  spec.param(request_queue_, "request_queue", "Request Queue", "Outgoing gRPC requests.");
  spec.param(allocator_, "allocator", "Allocator", "Output Allocator");

  cuda_stream_handler_.define_params(spec);
}

void GrpcClientRequestOp::compute(InputContext& op_input, OutputContext& op_output,
                                  ExecutionContext& context) {
  auto maybe_input_message = op_input.receive<holoscan::gxf::Entity>("input");
  if (!maybe_input_message) {
    HOLOSCAN_LOG_ERROR("grpc: Failed to receive input message");
    return;
  }
  auto request = std::make_shared<EntityRequest>();
  holoscan::ops::TensorProto::tensor_to_entity_request(
      maybe_input_message.value(), request, get_cuda_stream());
  request_queue_->push(request);
  HOLOSCAN_LOG_DEBUG("grpc: request converted and queued for transmission");
}

const cudaStream_t GrpcClientRequestOp::get_cuda_stream() {
  return cuda_stream_handler_.get_cuda_stream(fragment()->executor().context());
}

}  // namespace holoscan::ops
