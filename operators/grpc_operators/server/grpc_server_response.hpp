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

#ifndef GRPC_SERVER_RESPONSE_HPP
#define GRPC_SERVER_RESPONSE_HPP

#include <holoscan/utils/cuda_stream_handler.hpp>

#include "../common/conditional_variable_queue.hpp"
#include "../common/tensor_proto.hpp"
#include "holoscan.pb.h"

using holoscan::entity::EntityResponse;

namespace holoscan::ops {

/**
 * @class GrpcServerResponseOp
 * @brief A Holoscan Operator class for handling outgoing results.
 *
 * This class is Holoscan Operator that is responsible converting GXF Entities to EntityResponses
 * and queueing them for transmission to the remote client.
 *
 * The compute() method receives an GXF Entity, converts it to an EntityResponse (protobuf message),
 * and queues it for transmission.

 * ==Named Input==
 *
 * - **input** : holoscan::gxf::Entity
 *   - A GXF Entity with tensors to be transmitted to the remote client.
 *
 * ==Parameters==
 * - **response_queue** : A queue for storing outgoing gRPC results.
 */
class GrpcServerResponseOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcServerResponseOp)
  GrpcServerResponseOp() = default;

  void setup(OperatorSpec& spec) override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityResponse>>>>
      response_queue_;

  CudaStreamHandler cuda_stream_handler_;
};
}  // namespace holoscan::ops
#endif /* GRPC_SERVER_RESPONSE_HPP */
