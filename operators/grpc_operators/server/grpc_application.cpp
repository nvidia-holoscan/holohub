/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "grpc_application.hpp"

namespace holoscan::ops {

HoloscanGrpcApplication::HoloscanGrpcApplication(
    std::queue<std::shared_ptr<nvidia::gxf::Entity>>& incoming_request_queue,
    std::queue<std::shared_ptr<EntityResponse>>& outgoing_response_queue) {
  request_queue = make_resource<ConditionVariableQueue<std::shared_ptr<nvidia::gxf::Entity>>>(
      "request_queue", incoming_request_queue);
  response_queue = make_resource<ConditionVariableQueue<std::shared_ptr<EntityResponse>>>(
      "response_queue", outgoing_response_queue);
  streaming_enabled = make_condition<BooleanCondition>("streaming_enabled");
}

void HoloscanGrpcApplication::compose() {
  grpc_request_op = make_operator<GrpcServerRequestOp>(
      "grpc_request_op",
      streaming_enabled,
      make_condition<PeriodicCondition>("periodic-condition",
                                        Arg("recess_period") = std::string("60hz")),
      Arg("request_queue") = request_queue,
      Arg("allocator") = make_resource<UnboundedAllocator>("pool"),
      from_config("grpc_server"));

  grpc_response_op = make_operator<GrpcServerResponseOp>(
      "grpc_response_op", streaming_enabled, Arg("response_queue") = response_queue);
}

void HoloscanGrpcApplication::set_scheduler(const std::string& config_name) {
  scheduler(
      make_scheduler<holoscan::EventBasedScheduler>("event-scheduler", from_config(config_name)));
}

void HoloscanGrpcApplication::set_data_path(const std::string path) {
  data_path = path;
}

void HoloscanGrpcApplication::enqueue_request(const EntityRequest& request) {
  auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
      executor().context(), grpc_request_op->allocator()->gxf_cid());
  auto out_message = nvidia::gxf::Entity::New(executor().context());

  holoscan::ops::TensorProto::entity_request_to_tensor(
      &request, out_message.value(), gxf_allocator.value(), grpc_request_op->get_cuda_stream());

  auto entity = std::make_shared<nvidia::gxf::Entity>(out_message.value());
  request_queue->push(entity);
}

void HoloscanGrpcApplication::enqueue_response(std::shared_ptr<EntityResponse> response) {
  response_queue->push(response);
}

bool HoloscanGrpcApplication::is_response_available() {
  return !response_queue->empty();
}

std::shared_ptr<EntityResponse> HoloscanGrpcApplication::dequeue_response() {
  return response_queue->pop();
}

const uint32_t HoloscanGrpcApplication::rpc_timeout() {
  return grpc_request_op->rpc_timeout();
}

void HoloscanGrpcApplication::start_streaming() {
  streaming_enabled->enable_tick();
}

void HoloscanGrpcApplication::stop_streaming() {
  streaming_enabled->disable_tick();
}

}  // namespace holoscan::ops
