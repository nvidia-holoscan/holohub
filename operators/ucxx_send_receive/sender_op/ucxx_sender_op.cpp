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

#include "ucxx_sender_op.hpp"

#include <holoscan/holoscan.hpp>

#include <operators/ucxx_send_receive/serialize_tensor.hpp>

namespace holoscan::ops {

void UcxxSenderOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(tag_, "tag", "Tag", "UCX tag number", 0ul);
  spec.param(endpoint_, "endpoint", "Endpoint", "UcxxEndpoint resource");
  spec.param(max_in_flight_,
             "max_in_flight",
             "Max in-flight",
             "Maximum number of in-flight UCX send requests to retain. When exceeded, new inputs "
             "are dropped to bound memory retention. Defaults to 1.",
             1ul);
  spec.param(blocking_,
             "blocking",
             "Blocking",
             "If true, do not execute until endpoint is connected. If false (default), drain "
             "inputs and drop sends while disconnected (prevents upstream backpressure).",
             false);
  spec.input<holoscan::gxf::Entity>("in");

  bool blocking_value = false;
  for (auto arg : args()) {
    if (arg.name() == "blocking") {
      blocking_value = std::any_cast<bool>(arg.value());
      break;
    }
  }

  // If blocking == true: add the endpoint's is_alive_condition to this operator so that it will
  // only execute when the endpoint is alive.
  if (blocking_value) {
    for (auto arg : args()) {
      if (arg.name() == "endpoint") {
        auto resource = std::any_cast<std::shared_ptr<holoscan::Resource>>(arg.value());
        auto endpoint = std::dynamic_pointer_cast<UcxxEndpoint>(resource);
        if (endpoint) {
          add_arg(endpoint->is_alive_condition());
        } else {
          HOLOSCAN_LOG_ERROR("Failed to cast endpoint resource to UcxxEndpoint");
        }
        break;
      }
    }
  }
}

void UcxxSenderOp::stop() {
  for (auto& req : requests_) {
    if (req.header_request) { req.header_request->cancel(); }
    if (req.data_request) { req.data_request->cancel(); }
  }
}

void UcxxSenderOp::compute(holoscan::InputContext& input, holoscan::OutputContext&,
                           holoscan::ExecutionContext&) {
  auto in_message = input.receive<holoscan::gxf::Entity>("in").value();

  // Always clean up completed requests (even when disconnected).
  for (auto it = requests_.begin(); it != requests_.end();) {
    // Check if both header and data requests are completed
    bool header_done = !it->header_request || it->header_request->isCompleted();
    bool data_done = !it->data_request || it->data_request->isCompleted();
    if (!header_done || !data_done) {
      ++it;
      continue;
    }

    // Check status of both requests
    auto check_status = [](const std::shared_ptr<::ucxx::Request>& req, const char* name) {
      if (!req) return;
      if (ucs_status_t status = req->getStatus(); status != UCS_OK) {
        if (status == UCS_ERR_CONNECTION_RESET || status == UCS_ERR_NOT_CONNECTED ||
            status == UCS_ERR_UNREACHABLE || status == UCS_ERR_CANCELED) {
          HOLOSCAN_LOG_WARN("{} send failed (likely disconnect/reconnect) with status: {}",
                            name, ucs_status_string(status));
        } else {
          HOLOSCAN_LOG_ERROR("{} send failed with status: {}", name, ucs_status_string(status));
        }
      }
    };
    check_status(it->header_request, "Header");
    check_status(it->data_request, "Data");
    it = requests_.erase(it);
  }

  // Snapshot the UCXX endpoint for the duration of this tick.
  //
  // This avoids a race where the endpoint is reset concurrently (e.g., when the subscriber closes)
  // after the connectivity check but before tagSend() is invoked, which can otherwise lead to
  // calling tagSend() with a null endpoint.
  auto endpoint_resource = endpoint_.get();
  std::shared_ptr<::ucxx::Endpoint> ucxx_endpoint =
      endpoint_resource ? endpoint_resource->endpoint() : nullptr;

  // If not connected (waiting for subscriber to connect, or after disconnect),
  // drop the message to avoid stalling upstream operators like Holoviz.
  if (!ucxx_endpoint) {
    // Cancel any in-flight sends.
    for (auto& req : requests_) {
      if (req.cancel_requested) { continue; }
      if (req.header_request && !req.header_request->isCompleted()) {
        req.header_request->cancel();
      }
      if (req.data_request && !req.data_request->isCompleted()) { req.data_request->cancel(); }
      req.cancel_requested = true;
    }
    return;
  }

  // Bound outstanding buffer retention in case the network/receiver is slow.
  if (requests_.size() >= static_cast<size_t>(max_in_flight_.get())) {
    HOLOSCAN_LOG_WARN(
        "Dropping input: too many in-flight sends ({} >= max_in_flight={})",
        requests_.size(),
        max_in_flight_.get());
    return;
  }

  auto resolved = holoscan::ops::ucxx::resolveEntityBuffer(in_message);
  if (!resolved) {
    HOLOSCAN_LOG_ERROR("No sendable buffer found in input message");
    return;
  }

  SendRequest& send = requests_.emplace_back();
  send.keepalive_entity = in_message;
  send.header = resolved->header;

  const uint64_t tag_base = tag_.get();
  send.header_request = ucxx_endpoint->tagSend(
      &send.header, sizeof(holoscan::ops::ucxx::TensorHeader), ::ucxx::Tag{tag_base});
  send.data_request = ucxx_endpoint->tagSend(
      resolved->data_ptr, resolved->data_size, ::ucxx::Tag{tag_base + 1});
}

}  // namespace holoscan::ops
