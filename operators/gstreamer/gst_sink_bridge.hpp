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

#ifndef GST_SINK_BRIDGE_HPP
#define GST_SINK_BRIDGE_HPP

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>

#include "gst/object.hpp"
#include "gst/buffer.hpp"
#include "gst/caps.hpp"
#include "gst/video_info.hpp"

#include <holoscan/core/gxf/entity.hpp>

namespace holoscan {

// Forward declaration
class ExecutionContext;

/**
 * @brief Bridge between GStreamer and external data consumers
 *
 * This class provides a pure GStreamer implementation for managing an appsink element,
 * handling buffer retrieval from pipelines, and converting GStreamer buffers to GXF entities.
 * It contains no Holoscan-specific dependencies (except for ExecutionContext and GXF types).
 */
class GstSinkBridge {
 public:
  /**
   * @brief Constructor - creates and initializes the GStreamer appsink element
   * @param name Optional name for the appsink element
   * @param caps Capabilities string (e.g., "video/x-raw,format=RGBA")
   * @param max_buffers Maximum number of buffers in queue
   * @param qos_enabled Enable Quality of Service (frame dropping)
   * @throws std::runtime_error if initialization fails
   */
  GstSinkBridge(const std::string& name, const std::string& caps, 
                size_t max_buffers, bool qos_enabled);

  /**
   * @brief Destructor - cleans up GStreamer resources
   */
  ~GstSinkBridge();

  // Non-copyable and non-movable
  GstSinkBridge(const GstSinkBridge&) = delete;
  GstSinkBridge& operator=(const GstSinkBridge&) = delete;
  GstSinkBridge(GstSinkBridge&&) = delete;
  GstSinkBridge& operator=(GstSinkBridge&&) = delete;

  /**
   * @brief Get the underlying GStreamer element
   * @return Reference to the GStreamer element wrapper (appsink)
   */
  const gst::Element& get_gst_element() const {
    return sink_element_;
  }

  /**
   * @brief Asynchronously pop the next buffer from the GStreamer pipeline
   * @return Future that will be fulfilled when a buffer becomes available
   */
  std::future<gst::Buffer> pop_buffer();

  /**
   * @brief Create a TensorMap from GStreamer buffer with zero-copy
   * 
   * Supports both packed formats (RGBA, RGB) and planar formats (I420, NV12).
   * For multi-plane formats, creates separate tensors with naming convention:
   *   - "video_frame" for Y/luma plane
   *   - "video_frame_u", "video_frame_v" for chroma planes (I420)
   *   - "video_frame_uv" for interleaved chroma (NV12)
   * 
   * @param buffer GStreamer buffer containing the data
   * @return TensorMap containing one or more named tensors, empty map on failure
   */
  TensorMap create_tensor_map_from_buffer(const gst::Buffer& buffer) const;

  /**
   * @brief Get the current negotiated caps from the sink
   * @return Caps with automatic reference counting
   */
  gst::Caps get_caps() const;

 private:
  // Friend declarations for appsink callback functions
  friend void appsink_eos_callback(::GstAppSink* appsink, gpointer user_data);
  friend ::GstFlowReturn appsink_new_preroll_callback(::GstAppSink* appsink, gpointer user_data);
  friend ::GstFlowReturn appsink_new_sample_callback(::GstAppSink* appsink, gpointer user_data);

  // Configuration
  std::string name_;
  std::string caps_;
  bool qos_enabled_;
  size_t max_buffers_;

  // GStreamer element (appsink)
  gst::Element sink_element_;

  // Buffer queue for thread-safe async processing
  std::queue<gst::Buffer> buffer_queue_;
  // Single pending buffer pop request (only one request at a time)
  std::optional<std::promise<gst::Buffer>> pending_request_;
  mutable std::mutex mutex_;
  std::condition_variable queue_cv_;
  bool is_shutting_down_ = false;
};

}  // namespace holoscan

#endif /* GST_SINK_BRIDGE_HPP */

