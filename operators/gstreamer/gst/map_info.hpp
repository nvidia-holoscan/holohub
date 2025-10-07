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

#ifndef GST_MAP_INFO_HPP
#define GST_MAP_INFO_HPP

#include <gst/gst.h>
#include "buffer.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for GstBuffer memory mapping
 *
 * Automatically maps GstBuffer memory on construction and unmaps on destruction.
 * This ensures safe access to buffer data and prevents memory leaks from forgotten
 * unmap calls. Supports move semantics but prevents copying to avoid double unmapping.
 */
class MapInfo {
public:
  /**
   * @brief Constructor that maps the buffer
   * @param buffer Buffer to map (maintains reference during mapping)
   * @param flags Map flags (GST_MAP_READ, GST_MAP_WRITE, GST_MAP_READWRITE)
   * @throws std::runtime_error if buffer mapping fails
   */
  MapInfo(const Buffer& buffer, ::GstMapFlags flags);

  /**
   * @brief Destructor automatically unmaps the buffer
   */
  ~MapInfo();

  // Delete copy operations to prevent double unmapping
  MapInfo(const MapInfo&) = delete;
  MapInfo& operator=(const MapInfo&) = delete;

  // Allow move operations
  MapInfo(MapInfo&& other) noexcept;
  MapInfo& operator=(MapInfo&& other) noexcept;

  /**
   * @brief Check if mapping was successful
   * @return true if buffer is mapped and data is accessible
   */
  bool is_mapped() const { return mapped_; }

  /**
   * @brief Get pointer to mapped data
   * @return Pointer to buffer data, or nullptr if not mapped
   */
  guint8* data() const { return mapped_ ? gst_map_info_.data : nullptr; }

  /**
   * @brief Get size of mapped data
   * @return Size in bytes, or 0 if not mapped
   */
  gsize size() const { return mapped_ ? gst_map_info_.size : 0; }

  /**
   * @brief Get the native GStreamer GstMapInfo structure (advanced usage)
   * @return Reference to internal GstMapInfo
   */
  const ::GstMapInfo& native_map_info() const { return gst_map_info_; }

private:
  Buffer buffer_;      // Keep buffer alive during mapping
  ::GstMapInfo gst_map_info_;  // Native GStreamer structure
  bool mapped_;
};

}  // namespace gst
}  // namespace holoscan

#endif /* GST_MAP_INFO_HPP */

