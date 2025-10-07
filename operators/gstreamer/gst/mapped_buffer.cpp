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

#include "mapped_buffer.hpp"
#include <gst/video/video.h>
#include <sstream>

namespace holoscan {
namespace gst {

namespace {
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
}  // namespace

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

const Buffer& MappedBuffer::get_buffer() const {
  return buffer_;
}

}  // namespace gst
}  // namespace holoscan

