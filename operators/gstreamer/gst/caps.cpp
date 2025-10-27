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
#include <stdexcept>

namespace holoscan {
namespace gst {

// ============================================================================
// Caps Implementation - RAII for caps with member functions
// ============================================================================

Caps::Caps(::GstCaps* caps) : caps_(caps) {}

Caps::Caps(const std::string& caps_string) :
  caps_(gst_caps_from_string(caps_string.c_str())) {
  if (!caps_) {
    throw std::runtime_error("Invalid caps string: '" + caps_string + "'");
  }
} 

Caps::~Caps() {
  if (caps_)
    gst_caps_unref(caps_);
}

Caps::Caps(const Caps& other) : caps_(other.caps_) {
  if (caps_)
    gst_caps_ref(caps_);
}

Caps& Caps::operator=(const Caps& other) {
  if (this != &other) {
    // Clean up current caps
    if (caps_)
      gst_caps_unref(caps_);

    // Copy from other
    caps_ = other.caps_;
    if (caps_)
      gst_caps_ref(caps_);
  }
  return *this;
}

Caps::Caps(Caps&& other) noexcept : caps_(other.caps_) {
  other.caps_ = nullptr;
}

Caps& Caps::operator=(Caps&& other) noexcept {
  if (this != &other) {
    // Clean up current caps
    if (caps_)
      gst_caps_unref(caps_);

    // Move from other
    caps_ = other.caps_;
    other.caps_ = nullptr;
  }
  return *this;
}

::GstCaps* Caps::get() const {
  return caps_;
}

Caps::operator bool() const {
  return caps_ != nullptr;
}

::GstCaps* Caps::ref() const {
  if (caps_) {
    gst_caps_ref(caps_);
    return caps_;
  }
  return nullptr;
}

::GstCaps* Caps::release() {
  auto result = caps_;
  caps_ = nullptr;
  return result;
}

void Caps::reset() {
  if (caps_) {
    gst_caps_unref(caps_);
    caps_ = nullptr;
  }
}

const char* Caps::get_structure_name() const {
  if (caps_ == nullptr || get_size() == 0)
    return nullptr;

  ::GstStructure* structure = gst_caps_get_structure(caps_, 0);
  if (!structure)
    return nullptr;

  return gst_structure_get_name(structure);
}

std::optional<VideoInfo> Caps::get_video_info() const {
  const char* media_type = get_structure_name();
  if (!media_type || !g_str_has_prefix(media_type, "video/")) {
    return std::nullopt; // Not video caps
  }

  return VideoInfo(*this);
}

bool Caps::is_empty() const {
  // Note: GStreamer semantics - nullptr returns FALSE (not empty)
  // Empty caps is created with gst_caps_new_empty()
  return gst_caps_is_empty(caps_);
}

guint Caps::get_size() const {
  return gst_caps_get_size(caps_);
}

bool Caps::has_feature(const char* feature_name) const {
  if (!feature_name || caps_ == nullptr || get_size() == 0)
    return false;

  // Check if the caps contain the specified feature
  for (guint i = 0; i < get_size(); i++) {
    GstCapsFeatures* features = gst_caps_get_features(caps_, i);
    if (features && gst_caps_features_contains(features, feature_name))
      return true;
  }
  
  return false;
}

std::string Caps::to_string() const {
  gchar* caps_str = gst_caps_to_string(caps_);
  if (!caps_str) {
    return std::string();
  }
  std::string result(caps_str);
  g_free(caps_str);
  return result;
}

}  // namespace gst
}  // namespace holoscan

