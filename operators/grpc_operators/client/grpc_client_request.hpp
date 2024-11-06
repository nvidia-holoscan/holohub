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

#ifndef CLIENT_GRPC_CLIENT_REQUEST_HPP
#define CLIENT_GRPC_CLIENT_REQUEST_HPP

#include <gxf/app/graph_entity.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include "holoscan.pb.h"

#include "../common/asynchronous_condition_queue.hpp"
#include "../common/conditional_variable_queue.hpp"
#include "../common/tensor_proto.hpp"

using holoscan::entity::EntityRequest;
using holoscan::entity::EntityResponse;

namespace holoscan::ops {

/*
 * @class GrpcClientRequestOp
 * @brief A Holoscan Operator class for sending data the gRPC server.
 *
 * This class is Holoscan Operator that is responsible converting GXF Entities received from the
 * input, convert it to Protobuf EntityRequest and queue it for transmission to the server.
 *
 * The compute() method receives an GXF Entity, converts it to an EntityRequest (protobuf message)
 * and queues it for transmission.
 *
 * ==Named Inputs==
 *
 * - **input** : holoscan::gxf::Entity
 *   - A GXF Entity with tensors to be streamed to the remote server.
 *
 * ==Parameters==
 * - **request_queue** : A queue for storing outgoing gRPC requests.
 * - **allocator** : An allocator used when converting EntityResponse to GXF Entity.
 */
class GrpcClientRequestOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcClientRequestOp)
  GrpcClientRequestOp() = default;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

  std::shared_ptr<Allocator> allocator() { return allocator_.get(); }

  Executor& executor() { return fragment()->executor(); }

  const cudaStream_t get_cuda_stream();

 private:
  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>>> request_queue_;
  Parameter<std::shared_ptr<Allocator>> allocator_;

  CudaStreamHandler cuda_stream_handler_;
};
}  // namespace holoscan::ops
#endif /* CLIENT_GRPC_CLIENT_REQUEST_HPP */
