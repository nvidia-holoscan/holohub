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

#ifndef GST_COMMON_HPP
#define GST_COMMON_HPP

#include <memory>
#include <string>

#include <gst/gst.h>

namespace holoscan {

// ============================================================================
// RAII Wrappers for GStreamer Objects
// ============================================================================

/**
 * @brief RAII wrapper for GstBuffer using shared_ptr with automatic reference counting
 * 
 * This wrapper ensures proper reference counting for GstBuffer objects and provides
 * automatic cleanup when the last reference is released. The underlying buffer memory
 * is guaranteed to remain valid as long as any GstBufferGuard references it.
 */
using GstBufferGuard = std::shared_ptr<GstBuffer>;

/**
 * @brief RAII wrapper for GstCaps using shared_ptr with automatic reference counting
 * 
 * This wrapper ensures proper reference counting for GstCaps objects and provides
 * automatic cleanup when the last reference is released.
 */
using GstCapsGuard = std::shared_ptr<GstCaps>;

// ============================================================================
// Factory Functions for RAII Guards
// ============================================================================

/**
 * @brief Factory function to create GstBufferGuard with proper reference counting
 * 
 * Creates a shared_ptr wrapper around a GstBuffer that automatically handles
 * reference counting. The buffer is ref'd when the guard is created and unref'd
 * when the guard is destroyed.
 * 
 * @param buffer GstBuffer to wrap (can be nullptr)
 * @return GstBufferGuard with automatic reference counting, or nullptr if input was nullptr
 */
GstBufferGuard make_buffer_guard(GstBuffer* buffer);

/**
 * @brief Factory function to create GstCapsGuard with proper reference counting
 * 
 * Creates a shared_ptr wrapper around GstCaps that automatically handles
 * reference counting. The caps are ref'd when the guard is created and unref'd
 * when the guard is destroyed.
 * 
 * @param caps GstCaps to wrap (can be nullptr)
 * @return GstCapsGuard with automatic reference counting, or nullptr if input was nullptr
 */
GstCapsGuard make_caps_guard(GstCaps* caps);

// ============================================================================
// Helper Functions for GStreamer Analysis
// ============================================================================

/**
 * @brief Get media type string from caps
 * @param caps GstCaps to analyze
 * @return Media type string (e.g., "video/x-raw", "audio/x-raw") or nullptr if invalid
 */
const char* get_media_type_from_caps(GstCaps* caps);

/**
 * @brief Extract video format information from caps
 * @param caps GstCaps to analyze
 * @param width Output parameter for video width
 * @param height Output parameter for video height
 * @param format Output parameter for format string (optional, can be nullptr)
 * @return true if video information was extracted successfully
 */
bool get_video_info_from_caps(GstCaps* caps, int* width, int* height, const char** format = nullptr);

/**
 * @brief Extract audio format information from caps
 * @param caps GstCaps to analyze
 * @param channels Output parameter for number of audio channels
 * @param rate Output parameter for sample rate
 * @param format Output parameter for format string (optional, can be nullptr)
 * @return true if audio information was extracted successfully
 */
bool get_audio_info_from_caps(GstCaps* caps, int* channels, int* rate, const char** format = nullptr);

/**
 * @brief Get buffer metadata as a formatted string
 * @param buffer GstBuffer to analyze
 * @param caps Optional GstCaps for additional format information
 * @return Formatted string with buffer information (size, timestamps, etc.)
 */
std::string get_buffer_info_string(GstBuffer* buffer, GstCaps* caps = nullptr);

}  // namespace holoscan

#endif /* GST_COMMON_HPP */
