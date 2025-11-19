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

#pragma once

#include <any>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <typeindex>

#include "message_reflection.h"

namespace isaac {

// Associates Isaac messages with MessageReflection objects in order to perform
// message operations on flatbuffer messages stored in std::any.
//
// Notes:
// - Schema-name lookup is used by the replayer to pick the correct reflection for
//   deserialization. Schema names are mapped to type_index for type-safe lookup
//   through the message_reflection_map_.
// - Duplicate registrations are allowed and will overwrite previous mappings.
//   This ensures the latest reflection (and TableType) matches the buffers present
//   at runtime, which is important when the same message type is registered from
//   different translation units or during static initialization.
//
// This class is a singleton.
class MessageRegistry {
 public:
  static MessageRegistry& get_instance();

  template <typename NativeType, typename ReflectionType>
  void add_message() {
    auto reflection = std::make_unique<ReflectionType>();
    std::string schema_name(reflection->schema_name());  // Construct string from view
    std::type_index type_idx = typeid(std::shared_ptr<NativeType>);

    if (!schema_name.empty()) {
      // Map schema name to type_index for schema-based lookup.
      // Use insert_or_assign to ensure latest registration wins (inserts new or updates existing).
      schema_to_type_map_.insert_or_assign(schema_name, type_idx);
    }

    // Always overwrite type-to-reflection mapping to ensure latest registration wins
    message_reflection_map_[type_idx] = std::move(reflection);
  }

  std::optional<std::reference_wrapper<const MessageReflection>> get_message_reflection(
      const std::any& message) const;

  // Lookup by schema name used by the replayer for deserialization from MCAP.
  std::optional<std::reference_wrapper<const MessageReflection>> get_message_reflection_by_schema(
      const std::string& schema_name) const;

 private:
  MessageRegistry();
  ~MessageRegistry() = default;
  MessageRegistry(const MessageRegistry&) = delete;
  MessageRegistry& operator=(const MessageRegistry&) = delete;

  std::map<std::type_index, std::unique_ptr<MessageReflection>> message_reflection_map_;
  std::map<std::string, std::type_index> schema_to_type_map_;
};

}  // namespace isaac
