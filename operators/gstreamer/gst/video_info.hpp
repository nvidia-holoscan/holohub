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

#ifndef HOLOSCAN__GSTREAMER__GST__VIDEO_INFO_HPP
#define HOLOSCAN__GSTREAMER__GST__VIDEO_INFO_HPP

#include <gst/video/video.h>
#include "caps.hpp"

namespace holoscan {
namespace gst {

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

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__VIDEO_INFO_HPP */

