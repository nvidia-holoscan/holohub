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

#ifndef GRPC_CLIENT_RESPONSE_HPP
#define GRPC_CLIENT_RESPONSE_HPP

#include <gxf/app/graph_entity.hpp>
#include <holoscan/holoscan.hpp>

#include "../common/asynchronous_condition_queue.hpp"

namespace holoscan::ops {
using namespace std;

/**
 * @class GrpcClientResponseOp
 * @brief A Holoscan Operator class for passing data received from gRPC server to downstream
 * operators
 *
 * This class is a Holoscan Operator that is responsible for handling responses found in
 * the response_queue.
 *
 * The compute() method dequeues a GXF Entity from the response_queue and emits it to the output.
 *
 * ==Named Outputs==
 *
 * - **output** : holoscan::gxf::Entity
 *   - A GXF Entity with tensors to be transmitted to a downstream operator.
 *
 * ==Parameters==
 * - **response_queue** : A queue for storing incoming gRPC responses.
 * - **condition** : An Asynchronous Condition used to notify the operator when a new GXF Entity is
 *                   available in the response_queue.
 */
class GrpcClientResponseOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(GrpcClientResponseOp)

  GrpcClientResponseOp() = default;

  void start() override;

  void stop() override;

  void initialize() override;

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  Parameter<std::shared_ptr<AsynchronousCondition>> condition_;
  Parameter<std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>>>
      response_queue_;
};
}  // namespace holoscan::ops

#endif /* GRPC_CLIENT_RESPONSE_HPP */
