/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN__GSTREAMER__GST__CAPS_HPP
#define HOLOSCAN__GSTREAMER__GST__CAPS_HPP

#include <gst/gst.h>

#include <optional>
#include <string>

#include "mini_object.hpp"

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
class Caps : public MiniObjectBase<Caps, ::GstCaps> {
 public:
  /**
   * @brief Constructor from GstCaps pointer (takes ownership)
   * @param caps GstCaps to wrap - caller transfers ownership (nullptr is allowed)
   */
  explicit Caps(::GstCaps* caps = nullptr);

  /**
   * @brief Constructor from caps string
   * @param caps_string GStreamer caps string (e.g., "video/x-raw,format=RGBA")
   * @throws std::runtime_error if caps string is invalid
   */
  explicit Caps(const std::string& caps_string);

  /**
   * @brief Get the name of the first structure in the caps
   * @param index Index of the structure to get the name of (default is 0 - first/primary structure)
   * @return Structure name (e.g., "video/x-raw", "audio/x-raw") or nullptr if caps is
   * nullptr/empty/invalid
   */
  const char* get_structure_name(guint index = 0) const;

  /**
   * @brief Get the value of a specific field in a specific structure
   * @param fieldname Field name to get the value of (e.g., "framerate")
   * @param index Index of the structure to get the value from (default is 0 - first/primary
   * structure)
   * @return Value of the field, or nullptr if field not found or caps is nullptr/empty/invalid
   */
  const GValue* get_structure_value(const char* fieldname, guint index = 0) const;

  /**
   * @brief Get a structure from the caps at the specified index
   * @param index Index of the structure to get (default is 0 - first/primary structure)
   * @return Pointer to the GstStructure at the given index, or nullptr if caps is
   * nullptr/empty/invalid or index is out of bounds
   * @note The returned structure is owned by the caps and should not be freed
   */
  ::GstStructure* get_structure(guint index = 0) const;

  /**
   * @brief Extract video format information from caps
   * @return std::optional<VideoInfo> containing video information, or std::nullopt if not video
   * caps
   */
  std::optional<VideoInfo> get_video_info() const;

  /**
   * @brief Check if caps are empty (follows GStreamer semantics)
   * Note: Empty caps means gst_caps_new_empty(), NOT nullptr.
   * @return true if caps contain no structures, false if caps is nullptr or contains structures
   */
  bool is_empty() const;

  /**
   * @brief Get the number of structures in caps
   * @return Number of structures (0 for nullptr or empty caps)
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
   * @return String representation of the caps (GStreamer returns "NULL" for nullptr caps)
   */
  std::string to_string() const;

  static constexpr auto ref_func = gst_caps_ref;
  static constexpr auto unref_func = gst_caps_unref;
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__CAPS_HPP */
