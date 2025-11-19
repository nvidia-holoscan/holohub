/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ucxx_endpoint.h"

namespace isaac::os {

UcxxEndpoint::~UcxxEndpoint() { worker_->stopProgressThread(); }

void UcxxEndpoint::setup(holoscan::ComponentSpec& spec) {
  spec.param(hostname_, "hostname", "Hostname", "Hostname of the endpoint",
             std::string("127.0.0.1"));
  spec.param(port_, "port", "Port", "Port of the endpoint", 50008);
  spec.param(listen_, "listen", "Listen",
             "Whether to listen for connections (server), or initiate a connection (client)",
             false);

  is_alive_condition_ = fragment()->make_condition<holoscan::AsynchronousCondition>(
      fmt::format("{}_is_alive", name()));
}

void UcxxEndpoint::on_connection_request(ucp_conn_request_h conn_request) {
  endpoint_ =
      listener_->createEndpointFromConnRequest(conn_request, /*endpoint_error_handling=*/true);
  HOLOSCAN_LOG_INFO("Endpoint connected");

  // Mark operators ready to execute.
  is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);

  endpoint_->setCloseCallback(
      [this](ucs_status_t status, std::shared_ptr<void>) { on_endpoint_closed(status); }, nullptr);
}

void UcxxEndpoint::initialize() {
  if (is_initialized_) {
    return;
  }
  holoscan::Resource::initialize();

  context_ = ucxx::createContext({}, ucxx::Context::defaultFeatureFlags);
  worker_ = context_->createWorker();

  is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);

  if (listen_) {
    listener_ = worker_->createListener(
        port_,
        +[](ucp_conn_request_h conn_request, void* endpoint) {
          reinterpret_cast<UcxxEndpoint*>(endpoint)->on_connection_request(conn_request);
        },
        this);
    HOLOSCAN_LOG_INFO("Listening on: {}", port_);
  } else {
    endpoint_ = worker_->createEndpointFromHostname(hostname_, port_, true);
    HOLOSCAN_LOG_INFO("Connecting to: {}:{}", hostname_, port_);

    // Mark operators ready to execute.
    is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);

    endpoint_->setCloseCallback(
        [this](ucs_status_t status, std::shared_ptr<void>) { on_endpoint_closed(status); },
        nullptr);
  }

  worker_->startProgressThread(/*pollingMode=*/false);
}

void UcxxEndpoint::on_endpoint_closed(ucs_status_t status) {
  HOLOSCAN_LOG_INFO("Endpoint closed");
  if (status != UCS_OK) {
    HOLOSCAN_LOG_ERROR("Endpoint closed with status: {}", ucs_status_string(status));
  }

  // Prevent operators from executing until a new connection is established (server mode)
  // or indefinitely (client mode).
  if (listen_) {
    is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
  } else {
    is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
  }
}

};  // namespace isaac::os
