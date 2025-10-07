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

#include "video_info.hpp"

namespace holoscan {
namespace gst {

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

}  // namespace gst
}  // namespace holoscan

