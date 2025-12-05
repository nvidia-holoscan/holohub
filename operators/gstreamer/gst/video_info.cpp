/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "video_info.hpp"

#include <stdexcept>

#include "caps.hpp"

namespace holoscan {
namespace gst {

VideoInfo::VideoInfo() {
  gst_video_info_init(&video_info_);
}

VideoInfo::VideoInfo(const Caps& caps) {
  if (!gst_video_info_from_caps(&video_info_, caps.get()))
    throw std::runtime_error(
        "Failed to create VideoInfo from Caps - caps may not contain valid video information");
}

bool VideoInfo::set_format(GstVideoFormat format, gint width, gint height) {
  return gst_video_info_set_format(&video_info_, format, width, height);
}

gsize VideoInfo::get_size() const {
  return GST_VIDEO_INFO_SIZE(&video_info_);
}

gsize VideoInfo::get_stride(int plane_index) const {
  if (plane_index < 0 || static_cast<guint>(plane_index) >= video_info_.finfo->n_planes) {
    return 0;
  }
  return GST_VIDEO_INFO_PLANE_STRIDE(&video_info_, plane_index);
}

gsize VideoInfo::get_comp_height(int plane_index) const {
  if (plane_index < 0 || static_cast<guint>(plane_index) >= video_info_.finfo->n_planes) {
    return 0;
  }
  return GST_VIDEO_INFO_COMP_HEIGHT(&video_info_, plane_index);
}

gsize VideoInfo::get_comp_width(int plane_index) const {
  if (plane_index < 0 || static_cast<guint>(plane_index) >= video_info_.finfo->n_planes) {
    return 0;
  }
  return GST_VIDEO_INFO_COMP_WIDTH(&video_info_, plane_index);
}
gsize VideoInfo::get_size(int plane_index) const {
  if (plane_index < 0 || static_cast<guint>(plane_index) >= video_info_.finfo->n_planes) {
    return 0;
  }
  // Calculate plane size: stride * height (for this plane)
  return get_stride(plane_index) * get_comp_height(plane_index);
}

gsize VideoInfo::get_n_components() const {
  return GST_VIDEO_INFO_N_COMPONENTS(&video_info_);
}

gsize VideoInfo::get_n_planes() const {
  return GST_VIDEO_INFO_N_PLANES(&video_info_);
}

GstVideoFormat VideoInfo::get_format() const {
  return GST_VIDEO_INFO_FORMAT(&video_info_);
}

gint VideoInfo::get_width() const {
  return GST_VIDEO_INFO_WIDTH(&video_info_);
}

gint VideoInfo::get_height() const {
  return GST_VIDEO_INFO_HEIGHT(&video_info_);
}

}  // namespace gst
}  // namespace holoscan
