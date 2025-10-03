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

#include "gst_common.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief Holoscan Resource wrapper for GStreamer sink element
 *
 * This class provides a clean bridge between GStreamer pipelines and Holoscan operators.
 * The primary purpose is to enable Holoscan operators to retrieve and process data
 * from GStreamer pipelines.
 */
class SinkResource : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(SinkResource)
  using SharedPtr = std::shared_ptr<SinkResource>;

 /**
   * @brief Default constructor
   */
  SinkResource() = default;

  // Move semantics
  SinkResource(SinkResource&& other) noexcept = default;
  SinkResource& operator=(SinkResource&& other) noexcept = default;

  /**
   * @brief Destructor - cleans up GStreamer resources
   */
  ~SinkResource();

  /**
   * @brief Setup the resource parameters
   */
  void setup(holoscan::ComponentSpec& spec) override;

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
   * @brief Asynchronously get the next buffer from the GStreamer pipeline
   * @return Future that will be fulfilled when a buffer becomes available
   */
  std::future<holoscan::gst::MappedBuffer> get_buffer();

  /**
   * @brief Get the current negotiated caps from the sink
   * @return Caps with automatic reference counting
   */
  holoscan::gst::Caps get_caps() const;


  // Static member functions for GStreamer callbacks
  static ::GstCaps* get_caps_callback(::GstBaseSink *sink, ::GstCaps *filter);
  static gboolean set_caps_callback(::GstBaseSink *sink, ::GstCaps *caps);
  static ::GstFlowReturn render_callback(::GstBaseSink *sink, ::GstBuffer *buffer);
  static gboolean start_callback(::GstBaseSink *sink);
  static gboolean stop_callback(::GstBaseSink *sink);

 private:
  ::GstElement* sink_element_ = nullptr;

  // Buffer queue for thread-safe async processing
  std::queue<holoscan::gst::MappedBuffer> buffer_queue_;
  // Promise queue for pending buffer requests
  std::queue<std::promise<holoscan::gst::MappedBuffer>> request_queue_;
  mutable std::mutex mutex_;

  // Resource parameters
  holoscan::Parameter<std::string> caps_;
};

  using SinkResourcePtr = SinkResource::SharedPtr;

}  // namespace gst
}  // namespace holoscan

#endif /* GST_SINK_RESOURCE_HPP */
