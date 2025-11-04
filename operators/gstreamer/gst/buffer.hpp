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

#ifndef HOLOSCAN__GSTREAMER__GST__BUFFER_HPP
#define HOLOSCAN__GSTREAMER__GST__BUFFER_HPP

#include <memory>
#include <string>

#include <gst/gst.h>

#include "memory.hpp"
#include "object.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GstBuffer with automatic reference counting and member functions
 *
 * This class ensures proper reference counting for GstBuffer objects and provides
 * automatic cleanup when destroyed. It also provides convenient member functions
 * for common GstBuffer operations.
 * 
 * The constructors create a valid GstBuffer and throw std::runtime_error if
 * allocation fails, ensuring that all successfully constructed Buffer objects
 * are always in a valid state.
 */
class Buffer : public Object<::GstBuffer> {
public:
  /**
   * @brief Default constructor (creates empty buffer)
   * @throws std::runtime_error if buffer allocation fails
   */
  Buffer();

  /**
   * @brief Constructor from native GstBuffer
   * @param buffer Native GstBuffer pointer
   */
  explicit Buffer(::GstBuffer* buffer);

  /**
   * @brief Increment GStreamer reference count and return the raw pointer
   * @return Raw GstBuffer pointer
   */
  ::GstBuffer* ref() const override;

  /**
   * @brief Get buffer size in bytes
   * @return Buffer size in bytes
   */
  gsize size() const;

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
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__BUFFER_HPP */

