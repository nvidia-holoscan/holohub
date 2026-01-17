/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "ucxx_endpoint.hpp"

namespace holoscan::ops {

void UcxxEndpoint::add_close_callback(std::function<void(ucs_status_t)> callback) {
  std::scoped_lock lock(close_callbacks_mutex_);
  close_callbacks_.push_back(std::move(callback));
}

UcxxEndpoint::~UcxxEndpoint() {
  if (worker_) {
    worker_->stopProgressThread();
  }
}

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
  auto ep = listener_->createEndpointFromConnRequest(conn_request,
    /*endpoint_error_handling=*/true);
  std::atomic_store(&endpoint_, ep);
  HOLOSCAN_LOG_INFO("Endpoint connected");

  // Mark operators ready to execute.
  is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);

  ep->setCloseCallback(
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
    auto ep = worker_->createEndpointFromHostname(hostname_, port_, true);
    std::atomic_store(&endpoint_, ep);
    HOLOSCAN_LOG_INFO("Connecting to: {}:{}", hostname_, port_);

    // Mark operators ready to execute.
    is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_DONE);

    ep->setCloseCallback(
        [this](ucs_status_t status, std::shared_ptr<void>) { on_endpoint_closed(status); },
        nullptr);
  }

  worker_->startProgressThread(/*pollingMode=*/false);
}

void UcxxEndpoint::on_endpoint_closed(ucs_status_t status) {
  HOLOSCAN_LOG_INFO("Endpoint closed");
  if (status != UCS_OK) {
    // These are expected when subscriber disconnects/restarts.
    if (status == UCS_ERR_CONNECTION_RESET || status == UCS_ERR_NOT_CONNECTED ||
        status == UCS_ERR_UNREACHABLE || status == UCS_ERR_CANCELED) {
      HOLOSCAN_LOG_WARN("Endpoint closed (likely disconnect/reconnect) with status: {}",
                        ucs_status_string(status));
    } else {
      HOLOSCAN_LOG_ERROR("Endpoint closed with status: {}", ucs_status_string(status));
    }
  }

  // Notify any registered callbacks. (May be invoked from UCXX progress thread.)
  {
    std::vector<std::function<void(ucs_status_t)>> callbacks_copy;
    {
      std::scoped_lock lock(close_callbacks_mutex_);
      callbacks_copy = close_callbacks_;
    }
    for (auto& cb : callbacks_copy) {
      if (cb) { cb(status); }
    }
  }

  // Clear the endpoint so operators can quickly detect disconnection.
  std::atomic_store(&endpoint_, std::shared_ptr<::ucxx::Endpoint>{});

  // Prevent operators from executing until a new connection is established (server mode)
  // or indefinitely (client mode).
  if (listen_) {
    is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
  } else {
    is_alive_condition_->event_state(holoscan::AsynchronousEventState::EVENT_NEVER);
  }
}

};  // namespace holoscan::ops
