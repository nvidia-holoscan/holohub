/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "map_info.hpp"
#include <stdexcept>

namespace holoscan {
namespace gst {

// ============================================================================
// MapInfo Implementation - RAII for buffer memory mapping
// ============================================================================

MapInfo::MapInfo(const Buffer& buffer, ::GstMapFlags flags)
    : buffer_(buffer), mapped_(false) {
  mapped_ = gst_buffer_map(buffer_.get(), &gst_map_info_, flags);
  if (!mapped_) {
    throw std::runtime_error("Failed to map GstBuffer - buffer may be invalid or already mapped");
  }
}

MapInfo::~MapInfo() {
  if (mapped_) {
    gst_buffer_unmap(buffer_.get(), &gst_map_info_);
    mapped_ = false;
  }
}

MapInfo::MapInfo(MapInfo&& other) noexcept
    : buffer_(std::move(other.buffer_)), gst_map_info_(other.gst_map_info_), mapped_(other.mapped_) {
  other.buffer_ = Buffer();  // Move will leave it with empty buffer
  other.mapped_ = false;
}

MapInfo& MapInfo::operator=(MapInfo&& other) noexcept {
  if (this != &other) {
    // Clean up current mapping
    if (mapped_) {
      gst_buffer_unmap(buffer_.get(), &gst_map_info_);
    }

    // Move from other
    buffer_ = std::move(other.buffer_);
    gst_map_info_ = other.gst_map_info_;
    mapped_ = other.mapped_;

    // Reset other
    other.buffer_ = Buffer();  // Move will leave it with empty buffer
    other.mapped_ = false;
  }
  return *this;
}

}  // namespace gst
}  // namespace holoscan

