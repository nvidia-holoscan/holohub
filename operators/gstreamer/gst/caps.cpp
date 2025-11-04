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

#include <memory>
#include <stdexcept>

#include "video_info.hpp"

namespace holoscan {
namespace gst {

Caps::Caps(::GstCaps* caps) : Object<::GstCaps>(caps, [](::GstCaps* caps) {
  if (caps)
    gst_caps_unref(caps);
}) {}

Caps::Caps(const std::string& caps_string) :
  Caps(gst_caps_from_string(caps_string.c_str())) {
  if (!get()) {
    throw std::runtime_error("Invalid caps string: '" + caps_string + "'");
  }
} 

::GstCaps* Caps::ref() const {
  if (get())
    return gst_caps_ref(get());
  return nullptr;
}

const char* Caps::get_structure_name(guint index) const {
  if (!get() || index >= get_size())
    return nullptr;

  ::GstStructure* structure = gst_caps_get_structure(get(), index);
  if (!structure)
    return nullptr;

  return gst_structure_get_name(structure);
}

const GValue* Caps::get_structure_value(const char* fieldname, guint index) const {
  if (!get() || index >= get_size())
    return nullptr;

  ::GstStructure* structure = gst_caps_get_structure(get(), index);
  if (!structure)
    return nullptr;

  return gst_structure_get_value(structure, fieldname);
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
  return gst_caps_is_empty(get());
}

guint Caps::get_size() const {
  return gst_caps_get_size(get());
}

bool Caps::has_feature(const char* feature_name) const {
  if (!feature_name || get() == nullptr || get_size() == 0)
    return false;

  // Check if the caps contain the specified feature
  for (guint i = 0; i < get_size(); i++) {
    auto features = gst_caps_get_features(get(), i);
    if (gst_caps_features_contains(features, feature_name))
      return true;
  }
  return false;
}

std::string Caps::to_string() const {
  std::unique_ptr<gchar, decltype(&g_free)> caps_str(gst_caps_to_string(get()), &g_free);
  if (!caps_str)
    return std::string();
  return std::string(caps_str.get());
}

}  // namespace gst
}  // namespace holoscan

