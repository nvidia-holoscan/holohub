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

#include "buffer.hpp"

#include <stdexcept>

namespace holoscan {
namespace gst {

// ============================================================================
// Buffer Implementation - RAII for GstBuffer with member functions
// ============================================================================

gsize Buffer::get_size() const {
  if (!get())
    throw std::runtime_error("Invalid buffer");
  return gst_buffer_get_size(get());
}

GstBufferFlags Buffer::flags() const {
  if (!get())
    throw std::runtime_error("Invalid buffer");
  return static_cast<GstBufferFlags>(GST_BUFFER_FLAGS(get()));
}

void Buffer::append_memory(const gst::Memory& memory) {
  if (!get())
    throw std::runtime_error("Invalid buffer");
  // gst_buffer_append_memory takes ownership, so we need to give it a new reference
  gst_buffer_append_memory(get(), memory.ref().get());
}

guint Buffer::n_memory() const {
  if (!get())
    throw std::runtime_error("Invalid buffer");
  return gst_buffer_n_memory(get());
}

gst::Memory Buffer::get_memory(guint idx) const {
  if (!get())
    throw std::runtime_error("Invalid buffer");
  return gst::Memory(gst_buffer_get_memory(get(), idx));
}

bool Buffer::map(::GstMapInfo* info, ::GstMapFlags flags) const {
  if (!get())
    return false;
  return gst_buffer_map(get(), info, flags);
}

void Buffer::unmap(::GstMapInfo* info) const {
  if (get())
    gst_buffer_unmap(get(), info);
}

}  // namespace gst
}  // namespace holoscan
