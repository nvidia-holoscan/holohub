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

#ifndef GST_BUFFER_HPP
#define GST_BUFFER_HPP

#include <gst/gst.h>
#include <string>

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GstBuffer with automatic reference counting and member functions
 *
 * This class ensures proper reference counting for GstBuffer objects and provides
 * automatic cleanup when destroyed. It also provides convenient member functions
 * for common GstBuffer operations.
 * 
 * The default constructor creates an empty but valid GstBuffer, ensuring that
 * all Buffer objects are always in a valid state.
 */
class Buffer {
public:
  /**
   * @brief Default constructor (creates empty buffer)
   */
  Buffer();

  /**
   * @brief Constructor from native GstBuffer
   * @param buffer Native GstBuffer pointer (will be referenced)
   */
  explicit Buffer(::GstBuffer* buffer);

  /**
   * @brief Destructor (automatically unreferences the buffer)
   */
  ~Buffer();

  // Copy operations using GStreamer reference counting
  Buffer(const Buffer& other);
  Buffer& operator=(const Buffer& other);

  // Allow move operations
  Buffer(Buffer&& other) noexcept;
  Buffer& operator=(Buffer&& other) noexcept;

  /**
   * @brief Check if buffer is valid
   * @return true if buffer is valid (always true since we create a valid buffer)
   */
  bool is_valid() const { return true; }

  /**
   * @brief Get the underlying GstBuffer pointer
   * @return Native GstBuffer pointer (always valid)
   */
  ::GstBuffer* get() const { return buffer_; }

  /**
   * @brief Get buffer size in bytes
   * @return Buffer size in bytes
   */
  gsize size() const;

  /**
   * @brief Get presentation timestamp
   * @return PTS in nanoseconds, or GST_CLOCK_TIME_NONE if not set
   */
  GstClockTime pts() const;

  /**
   * @brief Get duration
   * @return Duration in nanoseconds, or GST_CLOCK_TIME_NONE if not set
   */
  GstClockTime duration() const;

  /**
   * @brief Get buffer flags
   * @return Buffer flags
   */
  GstBufferFlags flags() const;

private:
  ::GstBuffer* buffer_;
};

/**
 * @brief Get buffer metadata as a formatted string
 * @param buffer GstBuffer to analyze
 * @param caps Optional GstCaps for additional format information
 * @return Formatted string with buffer information (size, timestamps, etc.)
 */
std::string get_buffer_info_string(::GstBuffer* buffer, ::GstCaps* caps = nullptr);

}  // namespace gst
}  // namespace holoscan

#endif /* GST_BUFFER_HPP */

