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

#ifndef HOLOSCAN__GSTREAMER__GST__BUFFER_HPP
#define HOLOSCAN__GSTREAMER__GST__BUFFER_HPP

#include <gst/gst.h>

#include <memory>
#include <string>

#include "memory.hpp"
#include "mini_object.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GstBuffer with automatic reference counting and member functions
 *
 * This class ensures proper reference counting for GstBuffer objects and provides
 * automatic cleanup when destroyed. It also provides convenient member functions
 * for common GstBuffer operations.
 *
 * The constructor wraps an existing GstBuffer pointer (or nullptr). Most member functions
 * will throw std::runtime_error if called on an invalid (null) buffer. Exceptions:
 * - map() returns false on null buffer
 * - unmap() is a no-op on null buffer
 */
class Buffer : public MiniObjectBase<Buffer, ::GstBuffer> {
 public:
  /**
   * @brief Constructor from native GstBuffer
   * @param buffer Native GstBuffer pointer
   */
  explicit Buffer(::GstBuffer* buffer = nullptr) : MiniObjectBase(buffer) {}

  /**
   * @brief Create a new empty GStreamer buffer
   * @returns A new Buffer object wrapping a newly created GstBuffer
   * @note Wraps gst_buffer_new() for type-safe buffer creation
   */
  static Buffer create() { return Buffer(gst_buffer_new()); }

  /**
   * @brief Get buffer size in bytes
   * @return Buffer size in bytes
   */
  gsize get_size() const;

  /**
   * @brief Get buffer flags
   * @return Buffer flags
   */
  GstBufferFlags flags() const;

  /**
   * @brief Append a memory block to the buffer
   * @param memory Memory block to append
   */
  void append_memory(const gst::Memory& memory);

  /**
   * @brief Get the number of memory blocks in the buffer
   * @return Number of memory blocks
   */
  guint n_memory() const;

  /**
   * @brief Get a memory block from the buffer
   * @param idx Index of the memory block
   * @return Memory block at the specified index (with incremented refcount)
   */
  gst::Memory get_memory(guint idx) const;

  /**
   * @brief Map buffer for access
   * @param info Pointer to GstMapInfo structure to fill
   * @param flags GstMapFlags indicating the desired access mode
   * @return true if mapping was successful, false if buffer is null or mapping failed
   */
  bool map(::GstMapInfo* info, ::GstMapFlags flags) const;

  /**
   * @brief Unmap previously mapped buffer
   * @param info Pointer to GstMapInfo structure from the map call
   * @note This is a no-op if buffer is null
   */
  void unmap(::GstMapInfo* info) const;

  static constexpr auto ref_func = gst_buffer_ref;
  static constexpr auto unref_func = gst_buffer_unref;
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__BUFFER_HPP */
