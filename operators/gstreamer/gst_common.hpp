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
#include <optional>
#include <string>

#include <gst/gst.h>
#include <gst/video/video.h>

namespace holoscan {
namespace gst {

// Forward declarations
class Caps;
class VideoInfo;
class AudioInfo;

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
 * @brief RAII wrapper for GstCaps with automatic reference counting and member functions
 *
 * This class ensures proper reference counting for GstCaps objects and provides
 * automatic cleanup when destroyed. It also provides convenient member functions
 * for common GstCaps operations.
 */
class Caps {
public:
  /**
   * @brief Default constructor (creates empty caps)
   */
  Caps();

  /**
   * @brief Constructor from GstCaps pointer
   * @param caps GstCaps to wrap (if nullptr, creates empty caps)
   */
  explicit Caps(::GstCaps* caps);

  /**
   * @brief Destructor automatically unrefs the caps
   */
  ~Caps();

  // Copy operations using GStreamer reference counting
  Caps(const Caps& other);
  Caps& operator=(const Caps& other);

  // Allow move operations
  Caps(Caps&& other) noexcept;
  Caps& operator=(Caps&& other) noexcept;

  /**
   * @brief Check if caps are valid (always true since we use empty caps instead of null)
   * @return true (caps are always valid)
   */
  bool is_valid() const { return true; }

  /**
   * @brief Get the native GStreamer GstCaps pointer
   * @return Raw GstCaps pointer
   */
  ::GstCaps* get() const { return caps_; }

  /**
   * @brief Get media type string from caps
   * @return Media type string (e.g., "video/x-raw", "audio/x-raw") or nullptr if invalid
   */
  const char* get_media_type() const;

  /**
   * @brief Extract video format information from caps
   * @return std::optional<VideoInfo> containing video information, or std::nullopt if not video caps
   */
  std::optional<VideoInfo> get_video_info() const;

  /**
   * @brief Extract audio format information from caps
   * @return std::optional<AudioInfo> containing audio information, or std::nullopt if not audio caps
   */
  std::optional<AudioInfo> get_audio_info() const;

  /**
   * @brief Check if caps are empty
   * @return true if caps are empty
   */
  bool is_empty() const;

  /**
   * @brief Get the number of structures in caps
   * @return Number of structures
   */
  guint get_size() const;

private:
  ::GstCaps* caps_;
};

/**
 * @brief Video information extracted from GstCaps
 *
 * This class encapsulates video format information including dimensions and format.
 * It holds a private Caps object to ensure the underlying GstCaps remains valid.
 */
class VideoInfo {
public:
  /**
   * @brief Access the underlying GstVideoInfo structure directly
   * @return Pointer to GstVideoInfo structure
   */
  const ::GstVideoInfo* operator->() const { return &video_info_; }

private:
  /**
   * @brief Constructor from Caps object (private - only Caps can create VideoInfo)
   * @param caps Caps object containing video information
   */
  explicit VideoInfo(const Caps& caps);

  // Allow Caps class to create VideoInfo objects
  friend class Caps;

  Caps caps_;      // Keep caps alive to ensure GstCaps validity
  ::GstVideoInfo video_info_; // Cached GstVideoInfo for direct format access
};

/**
 * @brief Audio information extracted from GstCaps
 *
 * This class encapsulates audio format information including channels, sample rate, and format.
 * It holds a private Caps object to ensure the underlying GstCaps remains valid.
 */
class AudioInfo {
public:
  /**
   * @brief Get number of audio channels
   * @return Number of audio channels, or 0 if not available
   */
  int channels() const;

  /**
   * @brief Get audio sample rate
   * @return Sample rate in Hz, or 0 if not available
   */
  int rate() const;

  /**
   * @brief Get audio format string
   * @return Format string (e.g., "S16LE", "F32LE", "U8") or nullptr if not available
   */
  const char* format() const;

private:
  /**
   * @brief Constructor from Caps object (private - only Caps can create AudioInfo)
   * @param caps Caps object containing audio information
   */
  explicit AudioInfo(const Caps& caps);

  // Allow Caps class to create AudioInfo objects
  friend class Caps;

  Caps caps_;      // Keep caps alive to ensure GstCaps validity
  ::GstStructure* structure_; // Cached structure for efficient access
};

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
   * @param buffer GstBufferGuard to map (maintains reference during mapping)
   * @param flags Map flags (GST_MAP_READ, GST_MAP_WRITE, GST_MAP_READWRITE)
   */
  MapInfo(const GstBufferGuard& buffer, ::GstMapFlags flags);

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
GstBufferGuard make_buffer_guard(::GstBuffer* buffer);


// ============================================================================
// Helper Functions for GStreamer Analysis
// ============================================================================



/**
 * @brief Get buffer metadata as a formatted string
 * @param buffer GstBuffer to analyze
 * @param caps Optional GstCaps for additional format information
 * @return Formatted string with buffer information (size, timestamps, etc.)
 */
std::string get_buffer_info_string(::GstBuffer* buffer, ::GstCaps* caps = nullptr);

}  // namespace gst
}  // namespace holoscan

#endif /* GST_COMMON_HPP */
