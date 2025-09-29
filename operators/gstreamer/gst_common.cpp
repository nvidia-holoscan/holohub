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

// ============================================================================
// RAII Factory Function Implementations
// ============================================================================

GstBufferGuard make_buffer_guard(GstBuffer* buffer) {
    return buffer ? GstBufferGuard(gst_buffer_ref(buffer), gst_buffer_unref) : nullptr;
}

GstCapsGuard make_caps_guard(GstCaps* caps) {
    return caps ? GstCapsGuard(gst_caps_ref(caps), gst_caps_unref) : nullptr;
}

// ============================================================================
// GstMapInfo Implementation - RAII for buffer memory mapping
// ============================================================================

GstMapInfo::GstMapInfo(const GstBufferGuard& buffer, GstMapFlags flags)
    : buffer_(buffer), mapped_(false) {
  if (buffer_) {
    mapped_ = gst_buffer_map(buffer_.get(), &gst_map_info_, flags);
    if (!mapped_) {
      // Note: Using g_warning instead of HOLOSCAN_LOG_WARN to avoid dependency issues
      g_warning("Failed to map GstBuffer");
    }
  }
}

GstMapInfo::~GstMapInfo() {
  if (mapped_ && buffer_) {
    gst_buffer_unmap(buffer_.get(), &gst_map_info_);
    mapped_ = false;
  }
}

GstMapInfo::GstMapInfo(GstMapInfo&& other) noexcept
    : buffer_(std::move(other.buffer_)), gst_map_info_(other.gst_map_info_), mapped_(other.mapped_) {
  other.buffer_ = nullptr;  // Move will leave it empty, but this is explicit
  other.mapped_ = false;
}

GstMapInfo& GstMapInfo::operator=(GstMapInfo&& other) noexcept {
  if (this != &other) {
    // Clean up current mapping
    if (mapped_ && buffer_) {
      gst_buffer_unmap(buffer_.get(), &gst_map_info_);
    }
    
    // Move from other
    buffer_ = std::move(other.buffer_);
    gst_map_info_ = other.gst_map_info_;
    mapped_ = other.mapped_;
    
    // Reset other
    other.buffer_ = nullptr;  // Move will leave it empty, but this is explicit
    other.mapped_ = false;
  }
  return *this;
}

// ============================================================================
// Buffer and Caps Analysis Helper Functions
// ============================================================================

const char* get_media_type_from_caps(GstCaps* caps) {
  if (!caps || gst_caps_is_empty(caps) || gst_caps_get_size(caps) == 0) {
    return nullptr;
  }

  GstStructure* structure = gst_caps_get_structure(caps, 0);
  if (!structure) {
    return nullptr;
  }

  return gst_structure_get_name(structure);
}

bool get_video_info_from_caps(GstCaps* caps, int* width, int* height, const char** format) {
  if (!caps || !width || !height) {
    return false;
  }

  const char* media_type = get_media_type_from_caps(caps);
  if (!media_type || !g_str_has_prefix(media_type, "video/")) {
    return false;
  }

  GstStructure* structure = gst_caps_get_structure(caps, 0);
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

bool get_audio_info_from_caps(GstCaps* caps, int* channels, int* rate, const char** format) {
  if (!caps || !channels || !rate) {
    return false;
  }

  const char* media_type = get_media_type_from_caps(caps);
  if (!media_type || !g_str_has_prefix(media_type, "audio/")) {
    return false;
  }

  GstStructure* structure = gst_caps_get_structure(caps, 0);
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

std::string get_buffer_info_string(GstBuffer* buffer, GstCaps* caps) {
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

}  // namespace holoscan
