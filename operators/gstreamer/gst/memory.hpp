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

#ifndef HOLOSCAN__GSTREAMER__GST__MEMORY_HPP
#define HOLOSCAN__GSTREAMER__GST__MEMORY_HPP

#include <gst/gst.h>
#include "object.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GstMemory with automatic cleanup
 * 
 * This class manages the lifetime of a GstMemory object and automatically
 * calls gst_memory_unref when destroyed.
 */
class Memory : public Object<::GstMemory> {
 public:
  /**
   * @brief Constructor from raw pointer (takes ownership)
   * @param memory GstMemory pointer to wrap (nullptr is allowed)
   */
  explicit Memory(::GstMemory* memory = nullptr) 
    : Object(memory, [](::GstMemory* mem) {
        if (mem) {
          gst_memory_unref(mem);
        }
      }) {}

  /**
   * @brief Increment reference count and return the raw pointer
   * @return Raw GstMemory pointer with incremented refcount
   */
  ::GstMemory* ref() const override {
    if (get())
      return gst_memory_ref(get());
    return nullptr;
  }
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__MEMORY_HPP */

