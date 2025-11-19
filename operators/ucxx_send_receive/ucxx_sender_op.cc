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

#include "ucxx_sender_op.h"

#include "message_reflection.h"
#include "message_registry.h"
#include "tensor_materialization.h"

namespace isaac::os::ops {

void UcxxSenderOp::setup(holoscan::OperatorSpec& spec) {
  spec.param(tag_, "tag", "Tag", "UCX tag number", 0ul);
  spec.param(endpoint_, "endpoint", "Endpoint", "UcxxEndpoint resource");
  spec.input<std::any>("in");

  // Add the endpoint's is_alive_condition to this operator so that it will only execute when the
  // endpoint is alive.
  for (auto arg : args()) {
    if (arg.name() == "endpoint") {
      auto resource = std::any_cast<std::shared_ptr<holoscan::Resource>>(arg.value());
      auto endpoint = std::dynamic_pointer_cast<UcxxEndpoint>(resource);
      add_arg(endpoint->is_alive_condition());
      break;
    }
  }
}

void UcxxSenderOp::compute(holoscan::InputContext& input, holoscan::OutputContext&,
                           holoscan::ExecutionContext&) {
  auto message = input.receive<std::any>("in").value();

  // Look up the MessageReflection object for the message and use it to serialize the message to a
  // flatbuffer.
  const MessageRegistry& registry = MessageRegistry::get_instance();
  const std::optional<std::reference_wrapper<const MessageReflection>> reflection =
      registry.get_message_reflection(message);
  if (!reflection.has_value()) {
    HOLOSCAN_LOG_ERROR("Message type not registered");
    return;
  }
  SendRequest& send = requests_.emplace_back();
  {
    isaac::WithTensorMaterialization materialize_tensors;
    auto offset = reflection.value().get().pack(send.flatbuffer_builder, message);
    send.flatbuffer_builder.Finish(offset);
  }

  // Send the flatbuffer.
  send.request =
      endpoint_->endpoint()->tagSend(send.flatbuffer_builder.GetBufferPointer(),
                                     send.flatbuffer_builder.GetSize(), ucxx::Tag{tag_.get()});

  // Clean up completed requests.
  for (auto it = requests_.begin(); it != requests_.end();) {
    if (!it->request->isCompleted()) {
      it++;
      continue;
    }
    if (ucs_status_t status = it->request->getStatus(); status != UCS_OK) {
      HOLOSCAN_LOG_ERROR("Send failed with status: {}", ucs_status_string(status));
    }
    it = requests_.erase(it);
  }
}

}  // namespace isaac::os::ops
