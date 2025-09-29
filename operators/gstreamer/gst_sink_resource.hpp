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

#ifndef GST_SINK_RESOURCE_HPP
#define GST_SINK_RESOURCE_HPP

#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <holoscan/holoscan.hpp>

namespace holoscan {

// RAII wrapper for GstBuffer using shared_ptr with direct function deleter
using GstBufferGuard = std::shared_ptr<GstBuffer>;

// RAII wrapper for GstCaps using shared_ptr with direct function deleter
using GstCapsGuard = std::shared_ptr<GstCaps>;

/**
 * @brief Holoscan Resource wrapper for GStreamer sink element
 *
 * This class provides a clean bridge between GStreamer pipelines and Holoscan operators.
 * The primary purpose is to enable Holoscan operators to retrieve and process data
 * from GStreamer pipelines.
 */
class GstSinkResource : public holoscan::Resource {
 public:
 using SharedPtr = std::shared_ptr<GstSinkResource>;

 /**
   * @brief Default constructor
   */
  GstSinkResource() = default;

  /**
   * @brief Constructor with optional sink name
   * @param sink_name Name for the GStreamer element instance (optional)
   */
  explicit GstSinkResource(const std::string& sink_name = "")
      : sink_name_(sink_name) {}

  // Move semantics
  GstSinkResource(GstSinkResource&& other) noexcept = default;
  GstSinkResource& operator=(GstSinkResource&& other) noexcept = default;

  /**
   * @brief Destructor - cleans up GStreamer resources
   */
  ~GstSinkResource();

  /**
   * @brief Initialize the GStreamer sink resource
   */
  void initialize() override;

  /**
   * @brief Check if the resource is valid and ready to use
   * @return true if the sink element is created and ready
   */
  bool valid() const {
    return sink_element_ != nullptr;
  }

  /**
   * @brief Get the underlying GStreamer element
   * @return Pointer to the GstElement (do not unref manually)
   */
  GstElement* get_element() const {
    return sink_element_;
  }

  /**
   * @brief Get the sink name
   * @return The name of the sink element
   */
  const std::string& get_sink_name() const {
    return sink_name_;
  }

  /**
   * @brief Asynchronously get the next buffer from the GStreamer pipeline
   * @return Future that will be fulfilled when a buffer becomes available
   */
  std::future<GstBufferGuard> get_buffer();

  /**
   * @brief Get the current negotiated caps from the sink
   * @return GstCapsGuard with automatic reference counting, or nullptr if caps not negotiated yet
   */
  GstCapsGuard get_caps() const;

  // Static member functions for GStreamer callbacks
  static gboolean set_caps_callback(GstBaseSink *sink, GstCaps *caps);
  static GstFlowReturn render_callback(GstBaseSink *sink, GstBuffer *buffer);
  static gboolean start_callback(GstBaseSink *sink);
  static gboolean stop_callback(GstBaseSink *sink);

 private:
  std::string sink_name_;
  GstElement* sink_element_ = nullptr;

  // Buffer queue for thread-safe async processing
  std::queue<GstBufferGuard> buffer_queue_;
  // Promise queue for pending buffer requests
  std::queue<std::promise<GstBufferGuard>> request_queue_;
  mutable std::mutex mutex_;
};

using GstSinkResourcePtr = GstSinkResource::SharedPtr;

/**
 * @brief Helper functions for analyzing GStreamer buffers and caps
 */

/**
 * @brief Get media type string from caps
 * @param caps GstCaps to analyze
 * @return Media type string (e.g., "video/x-raw", "audio/x-raw") or nullptr
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

#endif /* GST_SINK_RESOURCE_HPP */
