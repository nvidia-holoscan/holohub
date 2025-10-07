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

#ifndef GST_MAPPED_BUFFER_HPP
#define GST_MAPPED_BUFFER_HPP

#include <gst/gst.h>
#include <string>
#include "buffer.hpp"
#include "video_info.hpp"
#include "map_info.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief RAII wrapper for mapped GstBuffer with automatic unmapping
 *
 * This class provides safe access to buffer data by automatically handling
 * the mapping and unmapping lifecycle. It ensures that buffers are properly
 * unmapped when the object goes out of scope, preventing memory leaks.
 */
class MappedBuffer {
public:
  /**
   * @brief Constructor that maps the buffer with video format information
   * @param buffer Buffer to map (maintains reference during mapping)
   * @param video_info VideoInfo containing format information
   * @param flags Map flags (GST_MAP_READ, GST_MAP_WRITE, GST_MAP_READWRITE)
   * @throws std::runtime_error if buffer mapping fails
   */
  MappedBuffer(const Buffer& buffer, const VideoInfo& video_info, ::GstMapFlags flags = GST_MAP_READ);

  // Move-only semantics (no copying to avoid double unmapping)
  MappedBuffer(const MappedBuffer&) = delete;
  MappedBuffer& operator=(const MappedBuffer&) = delete;

  MappedBuffer(MappedBuffer&& other) noexcept;
  MappedBuffer& operator=(MappedBuffer&& other) noexcept;

  /**
   * @brief Get pointer to mapped data
   * @return Pointer to buffer data
   */
  const guint8* data() const;

  /**
   * @brief Get size of mapped data
   * @return Size in bytes
   */
  gsize size() const;

  /**
   * @brief Get raw data pointer for a specific plane (video buffers only)
   * @param plane_index Plane index (0 for Y/luma, 1 for U/chroma, 2 for V/chroma)
   * @return Pointer to raw data for the specified plane, or nullptr if invalid
   */
  const guint8* get_plane_data(int plane_index) const;

  /**
   * @brief Get the VideoInfo associated with this buffer
   * @return Reference to VideoInfo object
   */
  const VideoInfo& get_video_info() const;

  /**
   * @brief Get reference to internal GstMapInfo
   * @return Reference to internal GstMapInfo
   */
  const MapInfo& get_map_info() const;

  /**
   * @brief Validate the mapped buffer data
   * @return true if buffer data is valid, false otherwise
   */
  bool validate() const;

  /**
   * @brief Get detailed validation report for this buffer
   * @return Detailed validation report string
   */
  std::string get_validation_report() const;

  /**
   * @brief Get the underlying Buffer object (for advanced usage)
   * @return Reference to the underlying Buffer object
   */
  const Buffer& get_buffer() const;

private:
  Buffer buffer_;           // Keep buffer alive during mapping
  VideoInfo video_info_;   // Video format information
  MapInfo map_info_;       // RAII wrapper for mapping
};

}  // namespace gst
}  // namespace holoscan

#endif /* GST_MAPPED_BUFFER_HPP */

