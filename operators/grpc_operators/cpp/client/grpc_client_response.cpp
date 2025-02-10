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

#include "grpc_client_response.hpp"

namespace holoscan::ops {

void GrpcClientResponseOp::start() {
  condition_->event_state(AsynchronousEventState::EVENT_WAITING);
}

void GrpcClientResponseOp::stop() {
  condition_->event_state(AsynchronousEventState::EVENT_NEVER);
  HOLOSCAN_LOG_INFO("grpc: GrpcClientResponseOp::stop()");
}

void GrpcClientResponseOp::initialize() {
  if (condition_.has_value()) { add_arg(condition_.get()); }
  Operator::initialize();
}

void GrpcClientResponseOp::setup(OperatorSpec& spec) {
  spec.param(response_queue_, "response_queue", "Response Queue", "Outgoing gRPC responses.");
  spec.param(condition_, "condition", "Asynchronous Condition", "Asynchronous Condition");

  spec.output<holoscan::gxf::Entity>("output");
}

void GrpcClientResponseOp::compute(InputContext& op_input, OutputContext& op_output,
                                   ExecutionContext& context) {
  std::shared_ptr<nvidia::gxf::Entity> response = response_queue_->pop();

  if (response) {
    auto result = nvidia::gxf::Entity(std::move(*response));
    op_output.emit(result, "output");
  }
  condition_->event_state(AsynchronousEventState::EVENT_WAITING);
}
}  // namespace holoscan::ops
