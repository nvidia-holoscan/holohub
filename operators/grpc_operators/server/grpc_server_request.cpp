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

#include "grpc_server_request.hpp"

namespace holoscan::ops {

void GrpcServerRequestOp::setup(OperatorSpec& spec) {
  spec.param(request_queue_, "request_queue", "Request Queue", "Incoming gRPC requests.");
  spec.param(allocator_,
             "allocator",
             "Allocator",
             "Allocator used for converting incoming EntityRequest objects to a GXF Entity.");
  spec.param(rpc_timeout_,
             "rpc_timeout",
             "RPC Call timeout",
             "Timeout in seconds for the gRPC server to issue a Finish command if no data is"
             "is transmitted or received.");

  spec.output<holoscan::gxf::Entity>("output");

  cuda_stream_handler_.define_params(spec);
}

void GrpcServerRequestOp::compute(InputContext& op_input, OutputContext& op_output,
                                  ExecutionContext& context) {
  if (!request_queue_->empty()) {
    auto request = request_queue_->pop();
    auto result = nvidia::gxf::Entity(std::move(*request));
    op_output.emit(result, "output");
  }
}

std::shared_ptr<UnboundedAllocator> GrpcServerRequestOp::allocator() {
  return allocator_.get();
}

const uint32_t GrpcServerRequestOp::rpc_timeout() {
  return rpc_timeout_.get();
}

const cudaStream_t GrpcServerRequestOp::get_cuda_stream() {
  return cuda_stream_handler_.get_cuda_stream(fragment()->executor().context());
}

}  // namespace holoscan::ops
