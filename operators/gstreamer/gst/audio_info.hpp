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

#ifndef GST_AUDIO_INFO_HPP
#define GST_AUDIO_INFO_HPP

#include <gst/gst.h>
#include "caps.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief Audio information extracted from GstCaps
 *
 * This class encapsulates audio format information including channels, sample rate, and format.
 * It holds a private Caps object to ensure the underlying GstCaps remains valid.
 */
class AudioInfo {
public:
  /**
   * @brief Get number of audio channels
   * @return Number of audio channels, or 0 if not available
   */
  int channels() const;

  /**
   * @brief Get audio sample rate
   * @return Sample rate in Hz, or 0 if not available
   */
  int rate() const;

  /**
   * @brief Get audio format string
   * @return Format string (e.g., "S16LE", "F32LE", "U8") or nullptr if not available
   */
  const char* format() const;

private:
  /**
   * @brief Constructor from Caps object (private - only Caps can create AudioInfo)
   * @param caps Caps object containing audio information
   */
  explicit AudioInfo(const Caps& caps);

  // Allow Caps class to create AudioInfo objects
  friend class Caps;

  Caps caps_;      // Keep caps alive to ensure GstCaps validity
  ::GstStructure* structure_; // Cached structure for efficient access
};

}  // namespace gst
}  // namespace holoscan

#endif /* GST_AUDIO_INFO_HPP */

