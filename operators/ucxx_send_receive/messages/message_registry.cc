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

#include "message_registry.h"

#include "tensor_generated.h"

namespace isaac {

MessageRegistry::MessageRegistry() {
  add_message<isaac::Tensor, DefaultMessageReflection<holoscan::Tensor, isaac::Tensor>>();
}

MessageRegistry& MessageRegistry::get_instance() {
  static MessageRegistry instance;
  return instance;
}

std::optional<std::reference_wrapper<const MessageReflection>>
MessageRegistry::get_message_reflection(const std::any& message) const {
  auto it = message_reflection_map_.find(message.type());
  if (it == message_reflection_map_.end()) {
    return std::nullopt;
  }
  return std::ref(*it->second);
}

std::optional<std::reference_wrapper<const MessageReflection>>
MessageRegistry::get_message_reflection_by_schema(const std::string& schema_name) const {
  // First lookup type_index by schema name.
  auto schema_it = schema_to_type_map_.find(schema_name);
  if (schema_it == schema_to_type_map_.end()) {
    return std::nullopt;
  }

  // Then lookup reflection by type_index.
  auto reflection_it = message_reflection_map_.find(schema_it->second);
  if (reflection_it == message_reflection_map_.end()) {
    return std::nullopt;
  }

  return std::ref(*reflection_it->second);
}

}  // namespace isaac
