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

#include <cassert>

namespace isaac {

// RAII scope object that manages thread-local state about whether to materialize tensors when
// packing them into flatbuffers.
//
// When an instance of this class exists, tensor data will be materialized (copied) into the
// flatbuffer. This is useful for scenarios where the tensor needs to be serialized with its
// actual data content rather than just metadata.
//
// TODO(mingxinz): If nesting scope is needed, we can use a reference counting mechanism.
//
// Usage:
//   {
//     WithTensorMaterialization scope;
//     // Any tensor packing operations in this scope will materialize data.
//     CreateTensor(builder, tensor_ptr);
//   }
//   // Outside the scope, tensors will not be materialized.
class WithTensorMaterialization {
 public:
  // Constructor enables tensor materialization for the current thread.
  WithTensorMaterialization() {
    assert(!enabled_);
    enabled_ = true;
  }

  // Destructor disables tensor materialization for the current thread.
  ~WithTensorMaterialization() { enabled_ = false; }

  // Returns whether tensor materialization is currently enabled for this thread.
  static bool enabled() { return enabled_; }

  // Delete copy and move constructors and assignment operators to ensure RAII semantics.
  WithTensorMaterialization(const WithTensorMaterialization&) = delete;
  WithTensorMaterialization& operator=(const WithTensorMaterialization&) = delete;
  WithTensorMaterialization(WithTensorMaterialization&&) = delete;
  WithTensorMaterialization& operator=(WithTensorMaterialization&&) = delete;

 private:
  inline static thread_local bool enabled_ = false;
};

}  // namespace isaac
