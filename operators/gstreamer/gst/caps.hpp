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

#ifndef GST_CAPS_HPP
#define GST_CAPS_HPP

#include <gst/gst.h>
#include <optional>
#include <string>

namespace holoscan {
namespace gst {

// Forward declarations
class VideoInfo;

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
   * @brief Check if caps are empty
   * @return true if caps are empty
   */
  bool is_empty() const;

  /**
   * @brief Get the number of structures in caps
   * @return Number of structures
   */
  guint get_size() const;

  /**
   * @brief Check if caps contain a specific feature
   * @param feature_name The feature name to check for (e.g., GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY)
   * @return true if any structure in the caps contains the specified feature
   */
  bool has_feature(const char* feature_name) const;

  /**
   * @brief Convert caps to human-readable string
   * @return String representation of the caps
   */
  std::string to_string() const;

private:
  ::GstCaps* caps_;
};

}  // namespace gst
}  // namespace holoscan

#endif /* GST_CAPS_HPP */

