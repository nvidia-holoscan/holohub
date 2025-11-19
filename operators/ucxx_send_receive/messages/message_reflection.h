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
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "flatbuffers/flatbuffers.h"
#include "flatbuffers/minireflect.h"
#include "flatbuffers/reflection.h"

namespace isaac {

// Base class to perform message operations on a flatbuffer.
class MessageReflection {
 public:
  virtual ~MessageReflection() = default;

  virtual std::vector<uint8_t> binary_schema() const = 0;
  virtual size_t binary_schema_size() const = 0;
  virtual std::string_view schema_name() const = 0;

  virtual const flatbuffers::TypeTable* type_table() const = 0;

  virtual flatbuffers::Offset<> pack(flatbuffers::FlatBufferBuilder& builder,
                                     const std::any& any_message) const = 0;

  virtual bool verify(const uint8_t* buffer, size_t size) const = 0;
  virtual const void* get_root(const uint8_t* buffer) const = 0;

  virtual std::any unpack(const uint8_t* buffer, size_t size) const = 0;
};

// Helper class to select the TableType by default.
template <typename T, typename = void>
struct DefaultTableType {
  using Type = T;
};
template <typename T>
struct DefaultTableType<T, std::void_t<typename T::TableType>> {
  using Type = typename T::TableType;
};

// Default instantiation of MessageReflection for a flatbuffer.
//
// Implementation notes:
// - We do not align incoming buffers because the FlatBuffers verifier and accessors already prevent
//   undefined behavior on unaligned input; copying for alignment is generally unnecessary on
//   modern platforms.
// - verify() guards against malformed or mismatched buffers before UnPackTo().
// - get_root() returns the correctly-typed table root used by UnPackTo().
template <typename NativeType, typename TableType = typename DefaultTableType<NativeType>::Type>
class DefaultMessageReflection : public MessageReflection {
 public:
  DefaultMessageReflection()
      : binary_schema_(TableType::BinarySchema::data()),
        binary_schema_size_(TableType::BinarySchema::size()),
        type_table_(TableType::MiniReflectTypeTable()) {}

  std::vector<uint8_t> binary_schema() const override {
    return std::vector<uint8_t>(binary_schema_, binary_schema_ + binary_schema_size_);
  }
  size_t binary_schema_size() const override { return binary_schema_size_; }

  std::string_view schema_name() const override {
    // Parse the binary schema using FlatBuffers reflection.
    auto schema = reflection::GetSchema(binary_schema_);
    if (!schema) {
      return "";  // Failed to parse schema.
    }

    // Get the root table.
    auto root_table = schema->root_table();
    if (!root_table) {
      return "";  // No root table found.
    }

    // Get the root table name.
    auto name = root_table->name();
    if (name) {
      return name->c_str();  // Direct view, no copy needed.
    }

    return "";  // Root table has no name.
  }

  const flatbuffers::TypeTable* type_table() const override { return type_table_; }

  flatbuffers::Offset<> pack(flatbuffers::FlatBufferBuilder& builder,
                             const std::any& any_message) const override {
    auto message_ptr = std::any_cast<const std::shared_ptr<NativeType>>(&any_message);
    if (!message_ptr) {
      throw std::runtime_error("Failed to cast std::any to std::shared_ptr<" +
                               std::string(typeid(NativeType).name()) +
                               "> in DefaultMessageReflection::pack(). " +
                               "Actual type held: " + std::string(any_message.type().name()));
    }
    return TableType::Pack(builder, message_ptr->get()).o;
  }

  bool verify(const uint8_t* buffer, size_t size) const override {
    flatbuffers::Verifier verifier(buffer, size);
    return verifier.VerifyBuffer<TableType>(nullptr);
  }

  const void* get_root(const uint8_t* buffer) const override {
    return flatbuffers::GetRoot<TableType>(buffer);
  }

  std::any unpack(const uint8_t* buffer, size_t size) const override {
    if (size < sizeof(flatbuffers::uoffset_t)) {
      throw std::runtime_error("Buffer too small for FlatBuffers.");
    }

    if (!verify(buffer, size)) {
      throw std::runtime_error("Buffer verification failed.");
    }

    auto root = get_root(buffer);
    auto table = static_cast<const TableType*>(root);
    if (!table) {
      throw std::runtime_error("Null table after cast.");
    }

    auto native = std::make_shared<NativeType>();
    table->UnPackTo(native.get());
    return std::any(native);
  }

 private:
  const uint8_t* binary_schema_;
  const size_t binary_schema_size_;
  const flatbuffers::TypeTable* type_table_;
};

}  // namespace isaac
