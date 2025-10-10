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

#include "buffer.hpp"
#include <holoscan/logger/logger.hpp>
#include <sstream>
#include <stdexcept>

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
}  // namespace

// ============================================================================
// Buffer Implementation - RAII for GstBuffer with member functions
// ============================================================================

Buffer::Buffer() : buffer_(gst_buffer_new()) {
  if (!buffer_) {
    throw std::runtime_error("Failed to create GStreamer buffer");
  }
}

Buffer::Buffer(::GstBuffer* buffer) : buffer_(buffer) {
  if (!buffer_) {
    throw std::runtime_error("Failed to create GStreamer buffer");
  }
}

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
  if (!other.buffer_) {
    // This should never happen, but if it does, we need to handle it
    // Can't throw in noexcept, so just log and terminate
    HOLOSCAN_LOG_CRITICAL("Failed to create GStreamer buffer in move constructor");
    std::terminate();
  }
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
  if (this != &other) {
    // Clean up current buffer
    gst_buffer_unref(buffer_);

    // Move from other
    buffer_ = other.buffer_;
    other.buffer_ = gst_buffer_new();
    if (!other.buffer_) {
      // This should never happen, but if it does, we need to handle it
      // Can't throw in noexcept, so just log and terminate
      HOLOSCAN_LOG_CRITICAL("Failed to create GStreamer buffer in move assignment");
      std::terminate();
    }
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
// Buffer Analysis Helper Functions
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

