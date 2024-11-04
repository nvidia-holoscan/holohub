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

#ifndef GRPC_SERVER_REQUEST_HPP
#define GRPC_SERVER_REQUEST_HPP

#include <gxf/app/graph_entity.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include "../common/conditional_variable_queue.hpp"

namespace holoscan::ops {
using namespace holoscan::ops;

/**
 * @class GrpcServerRequestOp
 * @brief A Holoscan Operator class for handling incoming gRPC requests.
 *
 * This class is Holoscan Operator that is responsible emitting GXF Entities from the request_queue
 * to the connected downstream operator(s).
 *
 * The compute() method checks the request_queue and emits the GXF Entity to the output port.

 * ==Named Outputs==
 *
 * - **output** : holoscan::gxf::Entity
 *   - A GXF Entity with tensors to be emitted to the downstream operator.
 *
 * ==Parameters==
 * - **request_queue** : A queue for storing incoming gRPC requests.
 * - **allocator** : An allocator used when converting EntityRequest to GXF Entity.
 * - **rpc_timeout** : Timeout in seconds for gRPC server to issue a Finish command if no data
 *                          is transmitted or received.
 */
class GrpcServerRequestOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcServerRequestOp)

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

  std::shared_ptr<UnboundedAllocator> allocator();

  const uint32_t rpc_timeout();

  const cudaStream_t get_cuda_stream();

 private:
  Parameter<std::shared_ptr<UnboundedAllocator>> allocator_;
  Parameter<uint32_t> rpc_timeout_;
  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>>>
      request_queue_;

  CudaStreamHandler cuda_stream_handler_;
};

}  // namespace holoscan::ops

#endif /* GRPC_SERVER_REQUEST_HPP */
