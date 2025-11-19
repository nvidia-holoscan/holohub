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

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace isaac {

// RAII scope to enable tensor gather mode with automatic size-based selection.
//
// The tensor size is determined by: byte_size = prod(shape) * (dtype.bits/8 * dtype.lanes)
//
// Usage examples:
//   {
//     WithTensorGather scope(1024);  // 1KB threshold
//     auto bytes = pack_image_message(msg);  // Automatic mode selection based on tensor size.
//   }
//
//
class WithTensorGather {
 public:
  explicit WithTensorGather(size_t size_threshold_bytes = 1024)
      : size_threshold_bytes_(size_threshold_bytes) {
    enabled_ = true;
    size_threshold_bytes_static_ = size_threshold_bytes;
  }
  ~WithTensorGather() {
    enabled_ = false;
    size_threshold_bytes_static_ = 0;
  }

  static bool enabled() { return enabled_; }
  static size_t size_threshold_bytes() { return size_threshold_bytes_static_; }

  WithTensorGather(const WithTensorGather&) = delete;
  WithTensorGather& operator=(const WithTensorGather&) = delete;

 private:
  size_t size_threshold_bytes_;
  inline static thread_local bool enabled_ = false;
  inline static thread_local size_t size_threshold_bytes_static_ = 0;
};

}  // namespace isaac
