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

#include "gst_common.hpp"

#include <sstream>
#include <gst/gst.h>

namespace holoscan {
namespace gst {

namespace {
// Helper function to extract media type from caps (internal use only)
const char* get_media_type_from_caps(::GstCaps* caps) {
  if (!caps || gst_caps_is_empty(caps) || gst_caps_get_size(caps) == 0) {
    return nullptr;
  }

  ::GstStructure* structure = gst_caps_get_structure(caps, 0);
  if (!structure) {
    return nullptr;
  }

  return gst_structure_get_name(structure);
}

// Internal validation functions (implementation details)
bool validate_buffer_size_internal(const ::GstVideoInfo* video_info, gsize buffer_size) {
  gsize expected_size = GST_VIDEO_INFO_SIZE(video_info);
  return buffer_size >= expected_size;
}

bool validate_stride_and_padding_internal(const ::GstVideoInfo* video_info, 
                                         const guint8* buffer_data, gsize buffer_size) {
  if (!buffer_data || buffer_size == 0) {
    return false;
  }

  // Check that we have enough data for all planes
  gsize total_expected_size = 0;
  for (int i = 0; i < video_info->finfo->n_planes; i++) {
    gsize stride = GST_VIDEO_INFO_PLANE_STRIDE(video_info, i);
    gsize height = GST_VIDEO_INFO_COMP_HEIGHT(video_info, i);
    gsize plane_size = stride * height;

    if (plane_size == 0) {
      return false; // Invalid plane size
    }
    total_expected_size += plane_size;
  }

  // Buffer should be at least as large as expected
  if (buffer_size < total_expected_size) {
    return false;
  }

  // For multi-plane formats, check that planes don't overlap
  if (video_info->finfo->n_planes > 1) {
    for (int i = 1; i < video_info->finfo->n_planes; i++) {
      gsize prev_stride = GST_VIDEO_INFO_PLANE_STRIDE(video_info, i - 1);
      gsize prev_height = GST_VIDEO_INFO_COMP_HEIGHT(video_info, i - 1);
      gsize prev_plane_size = prev_stride * prev_height;

      gsize current_plane_offset = GST_VIDEO_INFO_PLANE_OFFSET(video_info, i);
      gsize prev_plane_offset = GST_VIDEO_INFO_PLANE_OFFSET(video_info, i - 1);

      // Check for overlap
      if (current_plane_offset < prev_plane_offset + prev_plane_size) {
        return false;
      }
    }
  }

  return true;
}

bool detect_format_mismatch_internal(const ::GstVideoInfo* video_info,
                                   const guint8* buffer_data, gsize buffer_size) {
  if (!buffer_data || buffer_size == 0) {
    return false;
  }

  // Basic size validation
  if (!validate_buffer_size_internal(video_info, buffer_size)) {
    return false;
  }

  // Stride and padding validation
  if (!validate_stride_and_padding_internal(video_info, buffer_data, buffer_size)) {
    return false;
  }

  // For certain formats, we can do additional validation
  switch (video_info->finfo->format) {
    case GST_VIDEO_FORMAT_NV12:
    case GST_VIDEO_FORMAT_NV21:
      // YUV formats - check that UV planes are properly interleaved
      if (video_info->finfo->n_planes >= 2) {
        gsize y_stride = GST_VIDEO_INFO_PLANE_STRIDE(video_info, 0);
        gsize y_height = GST_VIDEO_INFO_COMP_HEIGHT(video_info, 0);
        gsize y_plane_size = y_stride * y_height;

        gsize uv_stride = GST_VIDEO_INFO_PLANE_STRIDE(video_info, 1);
        gsize uv_height = GST_VIDEO_INFO_COMP_HEIGHT(video_info, 1);
        gsize uv_plane_size = uv_stride * uv_height;

        // UV plane should be half the size of Y plane for NV12/NV21
        if (uv_plane_size != y_plane_size / 2) {
          return false;
        }
      }
      break;

    default:
      // For other formats, basic validation is sufficient
      break;
  }

  return true;
}

// Helper function to extract video format information from caps (internal use only)
bool get_video_info_from_caps(::GstCaps* caps, int* width, int* height, const char** format) {
  if (!caps || !width || !height) {
    return false;
  }

  const char* media_type = get_media_type_from_caps(caps);
  if (!media_type || !g_str_has_prefix(media_type, "video/")) {
    return false;
  }

  ::GstStructure* structure = gst_caps_get_structure(caps, 0);
  if (!structure) {
    return false;
  }

  if (!gst_structure_get_int(structure, "width", width) ||
      !gst_structure_get_int(structure, "height", height)) {
    return false;
  }

  if (format) {
    *format = gst_structure_get_string(structure, "format");
  }

  return true;
}

// Helper function to extract audio format information from caps (internal use only)
bool get_audio_info_from_caps(::GstCaps* caps, int* channels, int* rate, const char** format) {
  if (!caps || !channels || !rate) {
    return false;
  }

  const char* media_type = get_media_type_from_caps(caps);
  if (!media_type || !g_str_has_prefix(media_type, "audio/")) {
    return false;
  }

  ::GstStructure* structure = gst_caps_get_structure(caps, 0);
  if (!structure) {
    return false;
  }

  if (!gst_structure_get_int(structure, "channels", channels) ||
      !gst_structure_get_int(structure, "rate", rate)) {
    return false;
  }

  if (format) {
    *format = gst_structure_get_string(structure, "format");
  }

  return true;
}
} // unnamed namespace

// ============================================================================
// RAII Factory Function Implementations
// ============================================================================

// Note: make_buffer_guard function removed - use Buffer class constructor instead


// ============================================================================
// Buffer Implementation - RAII for GstBuffer with member functions
// ============================================================================

Buffer::Buffer() : buffer_(gst_buffer_new()) {}

Buffer::Buffer(::GstBuffer* buffer) : buffer_(buffer ? gst_buffer_ref(buffer) : gst_buffer_new()) {}

Buffer::~Buffer() {
  gst_buffer_unref(buffer_);
}

Buffer::Buffer(const Buffer& other) : buffer_(other.buffer_) {
  gst_buffer_ref(buffer_);
}

Buffer& Buffer::operator=(const Buffer& other) {
  if (this != &other) {
    // Clean up current buffer
    gst_buffer_unref(buffer_);

    // Copy from other
    buffer_ = other.buffer_;
    gst_buffer_ref(buffer_);
  }
  return *this;
}

Buffer::Buffer(Buffer&& other) noexcept : buffer_(other.buffer_) {
  other.buffer_ = gst_buffer_new();
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    // Clean up current buffer
    gst_buffer_unref(buffer_);

    // Move from other
    buffer_ = other.buffer_;
    other.buffer_ = gst_buffer_new();
  }
  return *this;
}

gsize Buffer::size() const {
  return gst_buffer_get_size(buffer_);
}

GstClockTime Buffer::pts() const {
  return GST_BUFFER_PTS(buffer_);
}

GstClockTime Buffer::duration() const {
  return GST_BUFFER_DURATION(buffer_);
}

GstBufferFlags Buffer::flags() const {
  return static_cast<GstBufferFlags>(GST_BUFFER_FLAGS(buffer_));
}

// ============================================================================
// MappedBuffer Implementation - RAII for buffer memory mapping
// ============================================================================

MappedBuffer::MappedBuffer(const Buffer& buffer, const VideoInfo& video_info, ::GstMapFlags flags)
    : buffer_(buffer), video_info_(video_info), map_info_(buffer, flags) {
}


MappedBuffer::MappedBuffer(MappedBuffer&& other) noexcept
    : buffer_(std::move(other.buffer_)), video_info_(std::move(other.video_info_)), map_info_(std::move(other.map_info_)) {
}

MappedBuffer& MappedBuffer::operator=(MappedBuffer&& other) noexcept {
  if (this != &other) {
    // Move from other
    buffer_ = std::move(other.buffer_);
    video_info_ = std::move(other.video_info_);
    map_info_ = std::move(other.map_info_);
  }
  return *this;
}

const guint8* MappedBuffer::data() const {
  return map_info_.data();
}

gsize MappedBuffer::size() const {
  return map_info_.size();
}

const guint8* MappedBuffer::get_plane_data(int plane_index) const {
  if (plane_index < 0 || plane_index >= video_info_.get()->finfo->n_planes) {
    return nullptr;
  }

  // Get the plane offset and return pointer to plane data
  const ::GstVideoInfo* gst_video_info = video_info_.get();
  gsize plane_offset = GST_VIDEO_INFO_PLANE_OFFSET(gst_video_info, plane_index);
  return map_info_.data() + plane_offset;
}

const MapInfo& MappedBuffer::get_map_info() const {
  return map_info_;
}

const VideoInfo& MappedBuffer::get_video_info() const {
  return video_info_;
}

bool MappedBuffer::validate() const {
  const guint8* buffer_data = data();
  gsize buffer_size = size();

  return detect_format_mismatch_internal(video_info_.get(), buffer_data, buffer_size);
}

std::string MappedBuffer::get_validation_report() const {
  const guint8* buffer_data = data();
  gsize buffer_size = size();
  const ::GstVideoInfo* video_info = video_info_.get();

  std::ostringstream report;

  report << "MappedBuffer Validation Report:\n";
  report << "  Format: " << gst_video_format_to_string(video_info->finfo->format) << "\n";
  report << "  Dimensions: " << video_info->width << "x" << video_info->height << "\n";
  report << "  Planes: " << video_info->finfo->n_planes << "\n";
  report << "  Expected total size: " << video_info_.get_total_size() << " bytes\n";
  report << "  Actual buffer size: " << buffer_size << " bytes\n";

  // Buffer size validation
  bool size_valid = validate_buffer_size_internal(video_info, buffer_size);
  report << "  Buffer size validation: " << (size_valid ? "PASS" : "FAIL") << "\n";

  // Stride and padding validation
  bool stride_valid = validate_stride_and_padding_internal(video_info, buffer_data, buffer_size);
  report << "  Stride/padding validation: " << (stride_valid ? "PASS" : "FAIL") << "\n";

  // Format mismatch detection
  bool format_valid = detect_format_mismatch_internal(video_info, buffer_data, buffer_size);
  report << "  Format mismatch detection: " << (format_valid ? "PASS" : "FAIL") << "\n";

  // Plane details
  report << "  Plane details:\n";
  for (int i = 0; i < video_info->finfo->n_planes; i++) {
    gsize stride = video_info_.get_plane_stride(i);
    gsize size = video_info_.get_plane_size(i);
    gsize offset = GST_VIDEO_INFO_PLANE_OFFSET(video_info, i);
    report << "    Plane " << i << ": stride=" << stride << ", size=" << size
           << ", offset=" << offset << "\n";
  }

  return report.str();
}

// ============================================================================
// VideoInfo Implementation
// ============================================================================

VideoInfo::VideoInfo(const Caps& caps) : caps_(caps) {
  // Extract GstVideoInfo for direct format access
  // No need to check media type - Caps::get_video_info() already validated it
  gst_video_info_from_caps(&video_info_, caps_.get());
}


gsize VideoInfo::get_total_size() const {
  return GST_VIDEO_INFO_SIZE(&video_info_);
}


gsize VideoInfo::get_plane_stride(int plane_index) const {
  if (plane_index < 0 || plane_index >= video_info_.finfo->n_planes) {
    return 0;
  }
  return GST_VIDEO_INFO_PLANE_STRIDE(&video_info_, plane_index);
}

gsize VideoInfo::get_plane_size(int plane_index) const {
  if (plane_index < 0 || plane_index >= video_info_.finfo->n_planes) {
    return 0;
  }
  // Calculate plane size: stride * height (for this plane)
  gsize stride = GST_VIDEO_INFO_PLANE_STRIDE(&video_info_, plane_index);
  gsize height = GST_VIDEO_INFO_COMP_HEIGHT(&video_info_, plane_index);
  return stride * height;
}


// ============================================================================
// AudioInfo Implementation
// ============================================================================

AudioInfo::AudioInfo(const Caps& caps) : caps_(caps), structure_(gst_caps_get_structure(caps_.get(), 0)) {
  // Extract and cache the GstStructure for efficient access
  // No need to check media type - Caps::get_audio_info() already validated it
}

int AudioInfo::channels() const {
  int channels = 0;
  gst_structure_get_int(structure_, "channels", &channels);
  return channels;
}

int AudioInfo::rate() const {
  int rate = 0;
  gst_structure_get_int(structure_, "rate", &rate);
  return rate;
}

const char* AudioInfo::format() const {
  return gst_structure_get_string(structure_, "format");
}

// ============================================================================
// Caps Implementation - RAII for caps with member functions
// ============================================================================

Caps::Caps() : caps_(gst_caps_new_empty()) {}

Caps::Caps(::GstCaps* caps) : caps_(caps ? gst_caps_ref(caps) : gst_caps_new_empty()) {}

Caps::~Caps() {
  gst_caps_unref(caps_);
}

Caps::Caps(const Caps& other) : caps_(other.caps_) {
  gst_caps_ref(caps_);
}

Caps& Caps::operator=(const Caps& other) {
  if (this != &other) {
    // Clean up current caps
    gst_caps_unref(caps_);

    // Copy from other
    caps_ = other.caps_;
    gst_caps_ref(caps_);
  }
  return *this;
}

Caps::Caps(Caps&& other) noexcept : caps_(other.caps_) {
  other.caps_ = gst_caps_new_empty();
}

Caps& Caps::operator=(Caps&& other) noexcept {
  if (this != &other) {
    // Clean up current caps
    gst_caps_unref(caps_);

    // Move from other
    caps_ = other.caps_;
    other.caps_ = gst_caps_new_empty();
  }
  return *this;
}

const char* Caps::get_media_type() const {
  return get_media_type_from_caps(caps_);
}

std::optional<VideoInfo> Caps::get_video_info() const {
  const char* media_type = get_media_type_from_caps(caps_);
  if (!media_type || !g_str_has_prefix(media_type, "video/")) {
    return std::nullopt; // Not video caps
  }

  return VideoInfo(*this);
}

std::optional<AudioInfo> Caps::get_audio_info() const {
  const char* media_type = get_media_type_from_caps(caps_);
  if (!media_type || !g_str_has_prefix(media_type, "audio/")) {
    return std::nullopt; // Not audio caps
  }

  return AudioInfo(*this);
}

bool Caps::is_empty() const {
  return gst_caps_is_empty(caps_);
}

guint Caps::get_size() const {
  return gst_caps_get_size(caps_);
}

// ============================================================================
// MapInfo Implementation - RAII for buffer memory mapping
// ============================================================================

MapInfo::MapInfo(const Buffer& buffer, ::GstMapFlags flags)
    : buffer_(buffer), mapped_(false) {
  mapped_ = gst_buffer_map(buffer_.get(), &gst_map_info_, flags);
  if (!mapped_) {
    throw std::runtime_error("Failed to map GstBuffer - buffer may be invalid or already mapped");
  }
}

MapInfo::~MapInfo() {
  if (mapped_) {
    gst_buffer_unmap(buffer_.get(), &gst_map_info_);
    mapped_ = false;
  }
}

MapInfo::MapInfo(MapInfo&& other) noexcept
    : buffer_(std::move(other.buffer_)), gst_map_info_(other.gst_map_info_), mapped_(other.mapped_) {
  other.buffer_ = Buffer();  // Move will leave it with empty buffer
  other.mapped_ = false;
}

MapInfo& MapInfo::operator=(MapInfo&& other) noexcept {
  if (this != &other) {
    // Clean up current mapping
    if (mapped_) {
      gst_buffer_unmap(buffer_.get(), &gst_map_info_);
    }

    // Move from other
    buffer_ = std::move(other.buffer_);
    gst_map_info_ = other.gst_map_info_;
    mapped_ = other.mapped_;

    // Reset other
    other.buffer_ = Buffer();  // Move will leave it with empty buffer
    other.mapped_ = false;
  }
  return *this;
}

// ============================================================================
// Buffer and Caps Analysis Helper Functions
// ============================================================================



std::string get_buffer_info_string(::GstBuffer* buffer, ::GstCaps* caps) {
  if (!buffer) {
    return "Invalid buffer";
  }

  std::stringstream info;
  info << "Buffer info: ";

  // Basic buffer information
  info << "size=" << gst_buffer_get_size(buffer) << " bytes";

  // Timestamp information
  if (GST_BUFFER_PTS_IS_VALID(buffer)) {
    info << ", pts=" << GST_BUFFER_PTS(buffer);
  }
  if (GST_BUFFER_DTS_IS_VALID(buffer)) {
    info << ", dts=" << GST_BUFFER_DTS(buffer);
  }
  if (GST_BUFFER_DURATION_IS_VALID(buffer)) {
    info << ", duration=" << GST_BUFFER_DURATION(buffer);
  }

  // Format information from caps
  if (caps) {
    const char* media_type = get_media_type_from_caps(caps);
    if (media_type) {
      info << ", type=" << media_type;

      if (g_str_has_prefix(media_type, "video/")) {
        int width, height;
        const char* format;
        if (get_video_info_from_caps(caps, &width, &height, &format)) {
          info << ", " << width << "x" << height;
          if (format) {
            info << " " << format;
          }
        }
      } else if (g_str_has_prefix(media_type, "audio/")) {
        int channels, rate;
        const char* format;
        if (get_audio_info_from_caps(caps, &channels, &rate, &format)) {
          info << ", " << channels << "ch " << rate << "Hz";
          if (format) {
            info << " " << format;
          }
        }
      }
    }
  }

  return info.str();
}

}  // namespace gst
}  // namespace holoscan
