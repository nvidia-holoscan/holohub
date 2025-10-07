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

#include "audio_info.hpp"

namespace holoscan {
namespace gst {

// ============================================================================
// AudioInfo Implementation
// ============================================================================

AudioInfo::AudioInfo(const Caps& caps) : caps_(caps), structure_(gst_caps_get_structure(caps_.get(), 0)) {
  // Extract and cache the GstStructure for efficient access
  // No need to check media type - Caps::get_audio_info() already validated it
}

int AudioInfo::channels() const {
  int channels = 0;
  gst_structure_get_int(structure_, "channels", &channels);
  return channels;
}

int AudioInfo::rate() const {
  int rate = 0;
  gst_structure_get_int(structure_, "rate", &rate);
  return rate;
}

const char* AudioInfo::format() const {
  return gst_structure_get_string(structure_, "format");
}

}  // namespace gst
}  // namespace holoscan

