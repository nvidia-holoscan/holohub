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

#ifndef GST_COMMON_HPP
#define GST_COMMON_HPP

#include <memory>
#include <string>

#include <gst/gst.h>

namespace holoscan {

// ============================================================================
// RAII Wrappers for GStreamer Objects
// ============================================================================

/**
 * @brief RAII wrapper for GstBuffer using shared_ptr with automatic reference counting
 *
 * This wrapper ensures proper reference counting for GstBuffer objects and provides
 * automatic cleanup when the last reference is released. The underlying buffer memory
 * is guaranteed to remain valid as long as any GstBufferGuard references it.
 */
using GstBufferGuard = std::shared_ptr<GstBuffer>;

/**
 * @brief RAII wrapper for GstCaps using shared_ptr with automatic reference counting
 *
 * This wrapper ensures proper reference counting for GstCaps objects and provides
 * automatic cleanup when the last reference is released.
 */
using GstCapsGuard = std::shared_ptr<GstCaps>;

/**
 * @brief RAII wrapper for GstBuffer memory mapping
 *
 * Automatically maps GstBuffer memory on construction and unmaps on destruction.
 * This ensures safe access to buffer data and prevents memory leaks from forgotten
 * unmap calls. Supports move semantics but prevents copying to avoid double unmapping.
 */
class GstMapInfo {
public:
  /**
   * @brief Constructor that maps the buffer
   * @param buffer GstBufferGuard to map (maintains reference during mapping)
   * @param flags Map flags (GST_MAP_READ, GST_MAP_WRITE, GST_MAP_READWRITE)
   */
  GstMapInfo(const GstBufferGuard& buffer, GstMapFlags flags);

  /**
   * @brief Destructor automatically unmaps the buffer
   */
  ~GstMapInfo();

  // Delete copy operations to prevent double unmapping
  GstMapInfo(const GstMapInfo&) = delete;
  GstMapInfo& operator=(const GstMapInfo&) = delete;

  // Allow move operations
  GstMapInfo(GstMapInfo&& other) noexcept;
  GstMapInfo& operator=(GstMapInfo&& other) noexcept;

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
  GstBufferGuard buffer_;      // Keep buffer alive during mapping
  ::GstMapInfo gst_map_info_;  // Native GStreamer structure
  bool mapped_;
};

// ============================================================================
// Factory Functions for RAII Guards
// ============================================================================

/**
 * @brief Factory function to create GstBufferGuard with proper reference counting
 *
 * Creates a shared_ptr wrapper around a GstBuffer that automatically handles
 * reference counting. The buffer is ref'd when the guard is created and unref'd
 * when the guard is destroyed.
 *
 * @param buffer GstBuffer to wrap (can be nullptr)
 * @return GstBufferGuard with automatic reference counting, or nullptr if input was nullptr
 */
GstBufferGuard make_buffer_guard(GstBuffer* buffer);

/**
 * @brief Factory function to create GstCapsGuard with proper reference counting
 *
 * Creates a shared_ptr wrapper around GstCaps that automatically handles
 * reference counting. The caps are ref'd when the guard is created and unref'd
 * when the guard is destroyed.
 *
 * @param caps GstCaps to wrap (can be nullptr)
 * @return GstCapsGuard with automatic reference counting, or nullptr if input was nullptr
 */
GstCapsGuard make_caps_guard(GstCaps* caps);

// ============================================================================
// Helper Functions for GStreamer Analysis
// ============================================================================

/**
 * @brief Get media type string from caps
 * @param caps GstCaps to analyze
 * @return Media type string (e.g., "video/x-raw", "audio/x-raw") or nullptr if invalid
 */
const char* get_media_type_from_caps(GstCaps* caps);

/**
 * @brief Extract video format information from caps
 * @param caps GstCaps to analyze
 * @param width Output parameter for video width
 * @param height Output parameter for video height
 * @param format Output parameter for format string (optional, can be nullptr)
 * @return true if video information was extracted successfully
 */
bool get_video_info_from_caps(GstCaps* caps, int* width, int* height, const char** format = nullptr);

/**
 * @brief Extract audio format information from caps
 * @param caps GstCaps to analyze
 * @param channels Output parameter for number of audio channels
 * @param rate Output parameter for sample rate
 * @param format Output parameter for format string (optional, can be nullptr)
 * @return true if audio information was extracted successfully
 */
bool get_audio_info_from_caps(GstCaps* caps, int* channels, int* rate, const char** format = nullptr);

/**
 * @brief Get buffer metadata as a formatted string
 * @param buffer GstBuffer to analyze
 * @param caps Optional GstCaps for additional format information
 * @return Formatted string with buffer information (size, timestamps, etc.)
 */
std::string get_buffer_info_string(GstBuffer* buffer, GstCaps* caps = nullptr);

}  // namespace holoscan

#endif /* GST_COMMON_HPP */
