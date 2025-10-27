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

#ifndef HOLOSCAN__GSTREAMER__GST__CAPS_HPP
#define HOLOSCAN__GSTREAMER__GST__CAPS_HPP

#include <optional>
#include <string>

#include <gst/gst.h>

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
   * @brief Constructor from GstCaps pointer (takes ownership, does NOT increment refcount)
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
   * @brief Get the native GStreamer GstCaps pointer
   * @return Raw GstCaps pointer
   */
   ::GstCaps* get() const;

  /**
   * @brief Check if caps pointer is not nullptr
   * @return true if caps_ is not nullptr, false otherwise
   */
  explicit operator bool() const;

  /**
   * @brief Increment GStreamer reference count and return the raw pointer
   * @return Raw GstCaps pointer
   */
  ::GstCaps* ref() const;

  /**
   * @brief Transfer ownership out of the guard (like std::unique_ptr::release)
   * Does NOT increment ref count - transfers the existing reference to the caller
   * @return Raw GstCaps pointer (caller takes ownership of the existing reference)
   */
  ::GstCaps* release();

  /**
   * @brief Reset the guard
   */
  void reset();

  /**
   * @brief Get the name of the first structure in the caps
   * @return Structure name (e.g., "video/x-raw", "audio/x-raw") or nullptr if caps is nullptr/empty/invalid
   */
  const char* get_structure_name() const;

  /**
   * @brief Extract video format information from caps
   * @return std::optional<VideoInfo> containing video information, or std::nullopt if not video caps
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

private:
  ::GstCaps* caps_;
};

}  // namespace gst
}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST__CAPS_HPP */

