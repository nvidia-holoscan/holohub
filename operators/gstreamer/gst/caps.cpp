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

#include "caps.hpp"
#include "video_info.hpp"
#include "audio_info.hpp"
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
}  // namespace

// ============================================================================
// Caps Implementation - RAII for caps with member functions
// ============================================================================

Caps::Caps() : caps_(gst_caps_new_empty()) {}

Caps::Caps(::GstCaps* caps) : caps_(caps ? gst_caps_ref(caps) : gst_caps_new_empty()) {}

Caps::Caps(const std::string& caps_string) {
  caps_ = gst_caps_from_string(caps_string.c_str());
  if (!caps_) {
    throw std::runtime_error("Invalid caps string: '" + caps_string + "'");
  }
}

Caps::~Caps() {
  gst_caps_unref(caps_);
}

Caps::Caps(const Caps& other) : caps_(other.caps_) {
  gst_caps_ref(caps_);
}

Caps& Caps::operator=(const Caps& other) {
  if (this != &other) {
    // Clean up current caps
    gst_caps_unref(caps_);

    // Copy from other
    caps_ = other.caps_;
    gst_caps_ref(caps_);
  }
  return *this;
}

Caps::Caps(Caps&& other) noexcept : caps_(other.caps_) {
  other.caps_ = gst_caps_new_empty();
}

Caps& Caps::operator=(Caps&& other) noexcept {
  if (this != &other) {
    // Clean up current caps
    gst_caps_unref(caps_);

    // Move from other
    caps_ = other.caps_;
    other.caps_ = gst_caps_new_empty();
  }
  return *this;
}

const char* Caps::get_media_type() const {
  return get_media_type_from_caps(caps_);
}

std::optional<VideoInfo> Caps::get_video_info() const {
  const char* media_type = get_media_type_from_caps(caps_);
  if (!media_type || !g_str_has_prefix(media_type, "video/")) {
    return std::nullopt; // Not video caps
  }

  return VideoInfo(*this);
}

std::optional<AudioInfo> Caps::get_audio_info() const {
  const char* media_type = get_media_type_from_caps(caps_);
  if (!media_type || !g_str_has_prefix(media_type, "audio/")) {
    return std::nullopt; // Not audio caps
  }

  return AudioInfo(*this);
}

bool Caps::is_empty() const {
  return gst_caps_is_empty(caps_);
}

guint Caps::get_size() const {
  return gst_caps_get_size(caps_);
}

bool Caps::has_feature(const char* feature_name) const {
  if (!feature_name || is_empty()) {
    return false;
  }

  guint caps_size = gst_caps_get_size(caps_);
  for (guint i = 0; i < caps_size; i++) {
    GstCapsFeatures* features = gst_caps_get_features(caps_, i);
    if (features && gst_caps_features_contains(features, feature_name)) {
      return true;
    }
  }
  
  return false;
}

}  // namespace gst
}  // namespace holoscan

