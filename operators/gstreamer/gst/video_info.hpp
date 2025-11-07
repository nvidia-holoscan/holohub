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

#ifndef HOLOSCAN__GSTREAMER__GST__VIDEO_INFO_HPP
#define HOLOSCAN__GSTREAMER__GST__VIDEO_INFO_HPP

#include <gst/video/video.h>

namespace holoscan {
namespace gst {

class Caps;

/**
 * @brief Video information extracted from GstCaps
 *
 * This class encapsulates video format information including dimensions and format.
 * GstVideoInfo is a POD structure that contains copied data from the caps, so it
 * does not hold references to the original GstCaps object.
 */
class VideoInfo {
 public:
  /**
   * @brief Constructor from Caps object
   * @param caps Caps object containing video information
   * @throws std::runtime_error if caps don't contain valid video information
   */
  explicit VideoInfo(const Caps& caps);

  /**
   * @brief Get total buffer size for all planes
   * @return Total size in bytes for all video planes
   */
  gsize get_size() const;

  /**
   * @brief Get stride (bytes per line) for a specific plane
   * @param plane_index Plane index
   * @return Stride in bytes, or 0 if invalid
   */
  gsize get_stride(int plane_index) const;

  /**
   * @brief Get component height for a specific plane
   * @param plane_index Plane index
   * @return Component height in pixels, or 0 if invalid
   */
  gsize get_comp_height(int plane_index) const;

  /**
   * @brief Get component width for a specific plane
   * @param plane_index Plane index
   * @return Component width in pixels, or 0 if invalid
   */
  gsize get_comp_width(int plane_index) const;

  /**
   * @brief Get plane size for a specific plane
   * @param plane_index Plane index
   * @return Size in bytes for the specified plane, or 0 if invalid
   */
  gsize get_size(int plane_index) const;

  /**
   * @brief Get number of components
   * @return Number of components, or 0 if invalid
   */
  gsize get_n_components() const;

  /**
   * @brief Get number of planes
   * @return Number of planes, or 0 if invalid
   */
  gsize get_n_planes() const;

  /**
   * @brief Get video format
   * @return GstVideoFormat enum value
   */
  GstVideoFormat get_format() const;

  /**
   * @brief Get width
   * @return Width in pixels
   */
  gint get_width() const;

  /**
   * @brief Get height
   * @return Height in pixels
   */
  gint get_height() const;

 private:
  ::GstVideoInfo video_info_;
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__VIDEO_INFO_HPP */
