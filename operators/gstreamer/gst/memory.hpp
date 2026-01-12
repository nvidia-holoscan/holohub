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

#ifndef HOLOSCAN__GSTREAMER__GST__MEMORY_HPP
#define HOLOSCAN__GSTREAMER__GST__MEMORY_HPP

#include <gst/gst.h>

#include "mini_object.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GstMemory with automatic cleanup
 *
 * This class manages the lifetime of a GstMemory object and automatically
 * calls gst_memory_unref when destroyed.
 */
class Memory : public MiniObjectBase<Memory, ::GstMemory> {
 public:
  /**
   * @brief Constructor from raw pointer (takes ownership)
   * @param memory GstMemory pointer to wrap (nullptr is allowed)
   */
  explicit Memory(::GstMemory* memory = nullptr) : MiniObjectBase(memory) {}

  /**
   * @brief Get the size of the memory
   * @param offset Pointer to store the offset (can be nullptr)
   * @param max_size Pointer to store the maxsize (can be nullptr)
   * @return The size of the memory in bytes
   */
  gsize get_sizes(gsize* offset = nullptr, gsize* max_size = nullptr) const;

  /**
   * @brief Map memory for access
   * @param info Pointer to GstMapInfo structure to fill
   * @param flags GstMapFlags indicating the desired access mode
   * @return true if mapping was successful, false otherwise
   */
  bool map(::GstMapInfo* info, ::GstMapFlags flags) const;

  /**
   * @brief Unmap previously mapped memory
   * @param info Pointer to GstMapInfo structure from the map call
   */
  void unmap(::GstMapInfo* info) const;

  /**
   * @brief Create a memory object that wraps existing data
   * @param flags Memory flags (e.g., GST_MEMORY_FLAG_READONLY)
   * @param data Pointer to the memory data to wrap
   * @param maxsize Maximum size of the memory
   * @param offset Offset within the memory
   * @param size Size of the accessible memory
   * @param user_data User data for the destroy callback
   * @param notify Callback to free user_data when memory is released
   * @returns A new Memory object wrapping the existing data
   * @note Wraps gst_memory_new_wrapped() for type-safe usage
   */
  static Memory create_wrapped(::GstMemoryFlags flags, void* data, gsize maxsize, gsize offset,
                               gsize size, void* user_data, ::GDestroyNotify notify) {
    return Memory(gst_memory_new_wrapped(flags, data, maxsize, offset, size, user_data, notify));
  }

  static constexpr auto ref_func = gst_memory_ref;
  static constexpr auto unref_func = gst_memory_unref;
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__MEMORY_HPP */
