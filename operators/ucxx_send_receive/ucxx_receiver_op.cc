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

#include "ucxx_receiver_op.h"

#include "message_registry.h"

namespace isaac::os::ops {

void UcxxReceiverOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(tag_, "tag", "Tag", "UCX tag number", 0ul);
  spec.param(schema_name_, "schema_name", "Schema name", "Schema name of received messages");
  spec.param(buffer_size_, "buffer_size", "Buffer size", "Receive buffer size", 4 << 10);
  spec.param(endpoint_, "endpoint", "Endpoint", "UcxxEndpoint resource");

  spec.output<std::any>("out");

  // Add the endpoint's is_alive_condition to this operator so that it will only execute only when
  // the endpoint is alive.
  for (auto arg : args()) {
    if (arg.name() == "endpoint") {
      auto resource = std::any_cast<std::shared_ptr<holoscan::Resource>>(arg.value());
      auto endpoint = std::dynamic_pointer_cast<UcxxEndpoint>(resource);
      add_arg(endpoint->is_alive_condition());
      break;
    }
  }
}

void UcxxReceiverOp::initialize() {
  holoscan::Operator::initialize();

  const MessageRegistry& registry = MessageRegistry::get_instance();
  reflection_ = registry.get_message_reflection_by_schema(schema_name_);
  if (!reflection_.has_value()) {
    throw std::runtime_error("Message schema not registered: " + schema_name_.get());
  }
}

void UcxxReceiverOp::start() { buffer_.resize(buffer_size_.get()); }

void UcxxReceiverOp::stop() {
  if (request_) {
    request_->cancel();
  }
}

void UcxxReceiverOp::compute([[maybe_unused]] holoscan::InputContext& input,
                             holoscan::OutputContext& output,
                             [[maybe_unused]] holoscan::ExecutionContext& context) {
  // If the receive request is complete, deserialize and emit it.
  if (request_ && request_->isCompleted()) {
    if (auto status = request_->getStatus(); status == UCS_OK) {
      std::any message = reflection_.value().get().unpack(buffer_.data(), buffer_.size());
      output.emit(message, "out");
    } else {
      HOLOSCAN_LOG_ERROR("Receive request failed with status: {}", ucs_status_string(status));
    }
    request_ = nullptr;
  }

  // Post a new request if none is pending.
  if (!request_) {
    async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    request_ = endpoint_->endpoint()->tagRecv(
        buffer_.data(), buffer_.size(), ucxx::Tag{tag_.get()}, ucxx::TagMaskFull,
        /*enablePythonFuture=*/false, [this](ucs_status_t, std::shared_ptr<void>) {
          async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
        });
  }
}

}  // namespace isaac::os::ops
