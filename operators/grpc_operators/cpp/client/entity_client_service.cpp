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

#include "entity_client_service.hpp"

namespace holoscan::ops {

EntityClientService::EntityClientService(
    const std::string& server_address, const uint32_t rpc_timeout, const bool interrupt,
    std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue,
    std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>>
        response_queue,
    std::shared_ptr<GrpcClientRequestOp> grpc_request_operator)
    : server_address_(server_address),
      rpc_timeout_(rpc_timeout),
      interrupt_(interrupt),
      request_queue_(request_queue),
      response_queue_(response_queue),
      grpc_request_operator_(grpc_request_operator) {}

void EntityClientService::start_entity_stream() {
  try {
    HOLOSCAN_LOG_DEBUG("grpc: Starting streaming client");
    entity_client_ = std::make_shared<EntityClient>(
        server_address_, rpc_timeout_, request_queue_, response_queue_);
    streaming_thread_ = std::thread(&EntityClientService::start_entity_stream_internal, this);
    HOLOSCAN_LOG_DEBUG("grpc: Entity client service configured: {}", server_address_);
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("grpc: Failed to connect to server at {}", server_address_);
    grpc_request_operator_->executor().interrupt();
  }
}

void EntityClientService::start_entity_stream_internal() {
  try {
    entity_client_->EntityStream(
        // Handle incoming responses
        [this](EntityResponse& response) {
          auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
              grpc_request_operator_->executor().context(),
              grpc_request_operator_->allocator()->gxf_cid());
          if (!gxf_allocator) { throw std::runtime_error("Failed to create GXF allocator"); }

          auto out_message = nvidia::gxf::Entity::New(grpc_request_operator_->executor().context());
          if (!out_message) { throw std::runtime_error("Failed to create GXF entity"); }

          holoscan::ops::TensorProto::entity_response_to_tensor(
              response,
              out_message.value(),
              gxf_allocator.value(),
              grpc_request_operator_->get_cuda_stream());
          return std::make_shared<nvidia::gxf::Entity>(out_message.value());
        },
        // Handle RPC completed event
        [this]() {});
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("grpc client: EntityStream rpc failed: {}", e.what());
  }
  entity_client_.reset();
  if (interrupt_) { grpc_request_operator_->executor().interrupt(); }
}

void EntityClientService::stop_entity_stream() {
  if (streaming_thread_.joinable()) { streaming_thread_.join(); }
  HOLOSCAN_LOG_INFO("grpc: Stopping streaming client");
}
}  // namespace holoscan::ops
