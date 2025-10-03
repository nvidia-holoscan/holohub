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
#include <stdexcept>
#include <string>

#include <gst/gst.h>
#include <gst/video/video.h>

namespace holoscan {
namespace gst {

// Forward declarations
class Iterator;
class Caps;
class VideoInfo;
class AudioInfo;
class Buffer;
class MapInfo;

// ============================================================================
// RAII Guards for GStreamer Objects
// ============================================================================

/**
 * @brief RAII wrapper for GstIterator with automatic cleanup and C++ iterator interface
 *
 * This class provides safe access to GStreamer iterators by automatically handling
 * the cleanup lifecycle and providing a C++-style iterator interface.
 */
class Iterator {
public:
  /**
   * @brief Constructor that takes ownership of a GstIterator
   * @param iterator GstIterator to wrap (takes ownership, can be nullptr)
   */
  explicit Iterator(::GstIterator* iterator);

  /**
   * @brief Destructor automatically frees the iterator
   */
  ~Iterator();

  // Delete copy operations to prevent double free
  Iterator(const Iterator&) = delete;
  Iterator& operator=(const Iterator&) = delete;

  // Allow move operations
  Iterator(Iterator&& other) noexcept;
  Iterator& operator=(Iterator&& other) noexcept;


  /**
   * @brief Advance iterator to next item (prefix increment)
   */
  void operator++();
  

  /**
   * @brief Get current item from iterator
   * @return Reference to current GValue (only valid when iterator is valid)
   */
  const ::GValue& operator*() const;

  
  /**
   * @brief Boolean conversion operator - check if current item is valid
   * @return true if current item can be safely accessed
   */
  explicit operator bool() const;



private:
  ::GstIterator* iterator_;
  ::GValue current_item_;
  ::GstIteratorResult last_result_;
};


/**
 * @brief RAII wrapper for GStreamer objects with automatic cleanup
 * @tparam T The GStreamer object type (GstElement, GstBus, etc.)
 */
template<typename T>
using GstObjectGuard = std::shared_ptr<T>;

/**
 * @brief Create a RAII guard for any GStreamer object that automatically calls gst_object_unref
 * @tparam T The GStreamer object type
 * @param object The GStreamer object to wrap (takes ownership)
 * @return Shared pointer that will automatically unref the object when destroyed
 */
template<typename T>
GstObjectGuard<T> make_gst_object_guard(T* object);

/**
 * @brief Convenience alias for GstElement guard
 */
using GstElementGuard = GstObjectGuard<GstElement>;

/**
 * @brief Convenience alias for GstBus guard
 */
using GstBusGuard = GstObjectGuard<GstBus>;

/**
 * @brief RAII wrapper for GstMessage with automatic cleanup
 */
using GstMessageGuard = std::shared_ptr<GstMessage>;

/**
 * @brief Create a RAII guard for a GstMessage that automatically calls gst_message_unref
 * @param message The GstMessage to wrap (takes ownership)
 * @return Shared pointer that will automatically unref the message when destroyed
 */
GstMessageGuard make_gst_message_guard(GstMessage* message);

/**
 * @brief RAII wrapper for GError with automatic cleanup
 */
using GstErrorGuard = std::shared_ptr<GError>;

/**
 * @brief Create a RAII guard for a GError that automatically calls g_error_free
 * @param error The GError to wrap (takes ownership)
 * @return Shared pointer that will automatically free the error when destroyed
 */
GstErrorGuard make_gst_error_guard(GError* error);

// ============================================================================
// RAII Wrappers for GStreamer Objects
// ============================================================================

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

// ============================================================================
// RAII Wrappers for GStreamer Objects
// ============================================================================

// Note: GstBufferGuard typedef removed - use Buffer class instead

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
   * @brief Constructor from caps string
   * @param caps_string GStreamer caps string (e.g., "video/x-raw,format=RGBA")
   * @throws std::runtime_error if caps string is invalid
   */
  explicit Caps(const std::string& caps_string);

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

  /**
   * @brief Get the underlying GstVideoInfo pointer
   * @return Native GstVideoInfo pointer
   */
  const ::GstVideoInfo* get() const { return &video_info_; }


  /**
   * @brief Get total buffer size for all planes
   * @return Total size in bytes for all video planes
   */
  gsize get_total_size() const;


  /**
   * @brief Get stride (bytes per line) for a specific plane
   * @param plane_index Plane index
   * @return Stride in bytes, or 0 if invalid
   */
  gsize get_plane_stride(int plane_index) const;

  /**
   * @brief Get plane size for a specific plane
   * @param plane_index Plane index
   * @return Size in bytes for the specified plane, or 0 if invalid
   */
  gsize get_plane_size(int plane_index) const;

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

private:
  Buffer buffer_;           // Keep buffer alive during mapping
  VideoInfo video_info_;   // Video format information
  MapInfo map_info_;       // RAII wrapper for mapping
};

// ============================================================================
// Factory Functions for RAII Guards
// ============================================================================

// Note: make_buffer_guard function removed - use Buffer class constructor instead


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
