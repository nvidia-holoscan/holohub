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

#include "entity_client.hpp"
#include <fmt/format.h>

namespace holoscan::ops {

EntityClient::EntityClient(
    const std::string& server_address, const uint32_t rpc_timeout,
    std::shared_ptr<ConditionVariableQueue<std::shared_ptr<EntityRequest>>> request_queue,
    std::shared_ptr<AsynchronousConditionQueue<std::shared_ptr<nvidia::gxf::Entity>>>
        response_queue)
    : rpc_timeout_(rpc_timeout), request_queue_(request_queue), response_queue_(response_queue) {
  channel_ = grpc::CreateChannel(server_address, grpc::InsecureChannelCredentials());
  if (auto status = channel_->GetState(true);
      status == GRPC_CHANNEL_TRANSIENT_FAILURE || status == GRPC_CHANNEL_SHUTDOWN) {
    throw std::runtime_error{"Error initializing channel. Please check the server address."};
  }
  stub_ = Entity::NewStub(channel_);
}

void EntityClient::EntityStream(on_new_response_available_callback response_cb,
                                on_rpc_completed_callback rpc_completed_cb) {
  EntityStreamInternal rpc_call(this, response_cb, rpc_completed_cb, rpc_timeout_);
  auto status = rpc_call.Await();
  if (status.ok()) {
    HOLOSCAN_LOG_INFO("grpc client: EntityStream rpc succeeded.");
  } else {
    throw std::runtime_error{
        fmt::format("EntityStream rpc failed with status: {}", status.error_message())};
  }
}

EntityClient::EntityStreamInternal::EntityStreamInternal(
    EntityClient* client, on_new_response_available_callback response_cb,
    on_rpc_completed_callback rpc_completed_cb, uint32_t rpc_timeout)
    : client_(client),
      response_cb_(response_cb),
      rpc_completed_cb_(rpc_completed_cb),
      rpc_timeout_(rpc_timeout) {
  last_network_activity_ = std::chrono::time_point<std::chrono::system_clock>::min();
  client_->stub_->async()->EntityStream(&context_, this);
  context_.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(rpc_timeout_));
  StartCall();
  Write();
  Read();
  writer_thread_ = std::thread(&EntityStreamInternal::ProcessOutgoingQueue, this);
}

EntityClient::EntityStreamInternal::~EntityStreamInternal() {
  writer_thread_.join();
}

void EntityClient::EntityStreamInternal::OnWriteDone(bool ok) {
  last_network_activity_ = std::chrono::high_resolution_clock::now();
  if (!ok) {
    HOLOSCAN_LOG_WARN("grpc client: write failed, error transmitting request");
    if (auto status = client_->channel_->GetState(true);
        status == GRPC_CHANNEL_TRANSIENT_FAILURE || status == GRPC_CHANNEL_SHUTDOWN) {
      HOLOSCAN_LOG_WARN("grpc client: closing connection");
      done_ = true;
      StartWritesDone();
    }
  }
  write_mutex_.unlock();
}

void EntityClient::EntityStreamInternal::OnReadDone(bool ok) {
  last_network_activity_ = std::chrono::high_resolution_clock::now();
  if (ok) {
    auto entity = response_cb_(response_);

    client_->response_queue_->push(entity);
    HOLOSCAN_LOG_DEBUG("grpc client: Response received and queued for display");
    Read();
  }
}
void EntityClient::EntityStreamInternal::OnDone(const grpc::Status& status) {
  status_ = status;
  done_ = true;
  rpc_completed_cb_();
  done_cv_.notify_one();
}

Status EntityClient::EntityStreamInternal::Await() {
  std::unique_lock<std::mutex> l(done_mutex_);
  done_cv_.wait(l, [this] { return done_; });
  return std::move(status_);
}

void EntityClient::EntityStreamInternal::Read() {
  response_.Clear();
  StartRead(&response_);
}

void EntityClient::EntityStreamInternal::Write() {
  if (!client_->request_queue_->empty()) {
    write_mutex_.lock();
    std::shared_ptr<EntityRequest> request;
    request = client_->request_queue_->pop();
    StartWrite(&*request);
    HOLOSCAN_LOG_DEBUG("grpc client: Sending request to server");
  }
}

void EntityClient::EntityStreamInternal::ProcessOutgoingQueue() {
  while (true) {
    if (network_timed_out()) {
      HOLOSCAN_LOG_INFO("grpc client: Connection timed out, closing connection");
      StartWritesDone();
      break;
    }
    Write();
  }
}

bool EntityClient::EntityStreamInternal::network_timed_out() {
  if (last_network_activity_ == std::chrono::time_point<std::chrono::system_clock>::min())
    return false;
  auto end = std::chrono::high_resolution_clock::now();
  auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(end - last_network_activity_);
  return elapsed.count() > rpc_timeout_;
}
}  // namespace holoscan::ops
