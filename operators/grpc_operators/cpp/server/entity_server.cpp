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

#include "entity_server.hpp"

namespace holoscan::ops {

HoloscanEntityServiceImpl::HoloscanEntityServiceImpl(
    on_new_entity_stream_rpc new_entity_stream_rpc,
    on_entity_stream_rpc_complete entity_stream_rpc_complete)
    : new_entity_stream_rpc_(new_entity_stream_rpc),
      entity_stream_rpc_complete_(entity_stream_rpc_complete) {}

grpc::ServerBidiReactor<EntityRequest, EntityResponse>* HoloscanEntityServiceImpl::EntityStream(
    CallbackServerContext* context) {
  HOLOSCAN_LOG_INFO("grpc server: EntityStreamInternal - new RPC received");
  std::queue<std::shared_ptr<nvidia::gxf::Entity>> incoming_request_queue;
  std::queue<std::shared_ptr<EntityResponse>> outgoing_response_queue;
  auto client = context->client_metadata();
  auto auth = context->auth_context();
  return new EntityStreamInternal(
      this,
      new_entity_stream_rpc_("EntityStream", incoming_request_queue, outgoing_response_queue),
      entity_stream_rpc_complete_);
}

HoloscanEntityServiceImpl::EntityStreamInternal::EntityStreamInternal(
    HoloscanEntityServiceImpl* server, std::shared_ptr<HoloscanGrpcApplication> app,
    on_entity_stream_rpc_complete entity_stream_rpc_complete)
    : server_(server), app_(app), entity_stream_rpc_complete_(entity_stream_rpc_complete) {
  if (app == nullptr) {
    Finish(grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED, "Resource occupied"));
  } else {
    last_network_activity_ = std::chrono::time_point<std::chrono::system_clock>::min();
    writer_thread_ = std::thread(&EntityStreamInternal::processOutgoingQueue, this);
    Write();
    Read();
  }
}

HoloscanEntityServiceImpl::EntityStreamInternal::~EntityStreamInternal() {
  if (writer_thread_.joinable()) { writer_thread_.join(); }
  HOLOSCAN_LOG_INFO("grpc server: RPC completed");
}

void HoloscanEntityServiceImpl::EntityStreamInternal::OnWriteDone(bool ok) {
  last_network_activity_ = std::chrono::high_resolution_clock::now();
  if (!ok) { HOLOSCAN_LOG_WARN("grpc server: write failed, error writing response"); }
  write_mutex_.unlock();
}

void HoloscanEntityServiceImpl::EntityStreamInternal::OnReadDone(bool ok) {
  last_network_activity_ = std::chrono::high_resolution_clock::now();
  if (ok) {
    app_->enqueue_request(request_);
    HOLOSCAN_LOG_DEBUG("grpc server: Request received and queued for processing");
    Read();
  } else {
    is_read_done_ = true;
  }
}

void HoloscanEntityServiceImpl::EntityStreamInternal::OnDone() {
  HOLOSCAN_LOG_DEBUG("grpc server: server streaming complete");
  entity_stream_rpc_complete_(app_);
  delete this;
}

void HoloscanEntityServiceImpl::EntityStreamInternal::Read() {
  request_.Clear();
  StartRead(&request_);
}

void HoloscanEntityServiceImpl::EntityStreamInternal::Write() {
  if (app_->is_response_available()) {
    write_mutex_.lock();
    std::shared_ptr<EntityResponse> response;
    response = app_->dequeue_response();
    StartWrite(&*response);
    HOLOSCAN_LOG_DEBUG("grpc server: Sending response to client");
  }
}

void HoloscanEntityServiceImpl::EntityStreamInternal::processOutgoingQueue() {
  while (true) {
    if (processing_timed_out()) {
      HOLOSCAN_LOG_DEBUG("grpc server: sending finish event");
      Finish(grpc::Status::OK);
      break;
    }
    Write();
  }
}

bool HoloscanEntityServiceImpl::EntityStreamInternal::processing_timed_out() {
  if (last_network_activity_ == std::chrono::time_point<std::chrono::system_clock>::min())
    return false;
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - last_network_activity_);
  return is_read_done_ && elapsed.count() > app_->rpc_timeout();
}

}  // namespace holoscan::ops
