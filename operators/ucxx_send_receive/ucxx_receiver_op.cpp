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

#include "ucxx_receiver_op.hpp"
#include "serialize_tensor.hpp"

#include <holoscan/holoscan.hpp>

namespace holoscan::ops {

void UcxxReceiverOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(tag_, "tag", "Tag", "UCX tag number", 0ul);
  spec.param(buffer_size_, "buffer_size", "Buffer size", "Receive buffer size", 4 << 10);
  spec.param(endpoint_, "endpoint", "Endpoint", "UcxxEndpoint resource");
  spec.param(allocator_, "allocator", "Allocator", "Allocator for staging buffers");
  spec.output<holoscan::gxf::Entity>("out");

  // Add the endpoint's is_alive_condition to this operator so that it will only execute only when
  // the endpoint is alive.
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
      auto gxf_tensor = holoscan::ops::ucxx::deserializeTensor(
        buffer_.data(), buffer_.size(), context.context(), allocator_.get().get());
      if (!gxf_tensor.has_value()) {
        HOLOSCAN_LOG_ERROR("Failed to deserialize tensor: {}", gxf_tensor.error().what());
        return;
      }

      // Create an entity and add the tensor as a component with name ""
      auto out_entity = holoscan::gxf::Entity::New(&context);

      // Add the GXF tensor to the entity
      auto tensor_handle =
          static_cast<nvidia::gxf::Entity&>(out_entity).add<nvidia::gxf::Tensor>("");
      if (!tensor_handle) {
        HOLOSCAN_LOG_ERROR("Failed to add tensor to entity");
        return;
      }

      // Move the deserialized tensor data into the entity's tensor
      *tensor_handle.value() = std::move(gxf_tensor.value());

      output.emit(out_entity, "out");
    } else {
      HOLOSCAN_LOG_ERROR("Receive request failed with status: {}", ucs_status_string(status));
    }
    request_ = nullptr;
  }

  // Post a new request if none is pending.
  if (!request_) {
    // Snapshot the UCXX endpoint for the duration of this tick to avoid races with disconnects.
    auto endpoint_resource = endpoint_.get();
    std::shared_ptr<::ucxx::Endpoint> ucxx_endpoint =
        endpoint_resource ? endpoint_resource->endpoint() : nullptr;
    if (!ucxx_endpoint) { return; }

    async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_WAITING);
    request_ = ucxx_endpoint->tagRecv(
        buffer_.data(), buffer_.size(), ::ucxx::Tag{tag_.get()}, ::ucxx::TagMaskFull,
        /*enablePythonFuture=*/false, [this](ucs_status_t, std::shared_ptr<void>) {
          async_condition()->event_state(holoscan::AsynchronousEventState::EVENT_DONE);
        });
  }
}

}  // namespace holoscan::ops
