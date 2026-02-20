/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef HOLOSCAN__GSTREAMER__GST_SINK_BRIDGE_HPP
#define HOLOSCAN__GSTREAMER__GST_SINK_BRIDGE_HPP

#include <gst/gst.h>

#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include <holoscan/core/domain/tensor_map.hpp>

#include "gst/app_sink.hpp"
#include "gst/buffer.hpp"
#include "gst/caps.hpp"
#include "gst/config.hpp"
#include "gst_wait_group.hpp"

namespace holoscan {

/**
 * @brief Bridge between GStreamer and external data consumers
 *
 * This class provides a GStreamer implementation for managing an appsink element,
 * handling buffer retrieval from GStreamer pipelines, and converting GStreamer buffers to tensors.
 * It has minimal Holoscan dependencies via TensorMap for tensor data representation.
 */
class GstSinkBridge {
 public:
  /**
   * @brief Constructor - creates and initializes the GStreamer appsink element
   * @param name Optional name for the appsink element
   * @param caps_string Capabilities string (e.g., "video/x-raw,format=RGBA")
   * @param max_buffers Maximum number of buffers in queue (0 = unlimited)
   * @param qos Enable Quality of Service (frame dropping)
   * @throws std::runtime_error if initialization fails
   */
  GstSinkBridge(const std::string& name, const std::string& caps_string,
                size_t max_buffers, bool qos);

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
   * @return GStreamer element wrapper (appsink)
   */
  gst::Element get_gst_element() const;

  /**
   * @brief Get a future that will be ready when EOS is received by the appsink
   *
   * The returned shared_future can be waited on with any timeout or blocking strategy.
   * Multiple callers can wait on the same future.
   *
   * @return Shared future that becomes ready when EOS is received
   *
   * @note The user is responsible for waiting on this future if they need to know when
   *       the stream ends. The destructor does NOT wait for EOS automatically.
   */
  std::shared_future<void> get_eos_future() const;

  /**
   * @brief Asynchronously pull the next buffer from the GStreamer pipeline
   * @return Future that will be fulfilled when a buffer becomes available
   */
  std::future<gst::Buffer> pull_buffer();

  /**
   * @brief Create a TensorMap from GStreamer buffer with zero-copy
   *
   * Supports both packed formats (RGBA, RGB) and planar formats (I420, NV12).
   * For multi-plane formats, creates separate tensors with naming convention:
   *   - "video_frame" for Y/luma plane
   *   - "video_frame_u", "video_frame_v" for chroma planes (I420)
   *   - "video_frame_uv" for interleaved chroma (NV12)
   *
   * @param buffer GStreamer buffer containing the data (passed by value for ownership)
   * @return TensorMap containing one or more named tensors, empty map on failure
   *
   * @note The buffer is captured by the tensor deleter to ensure it stays alive
   *       for the lifetime of the returned tensors.
   */
  TensorMap create_tensor_map_from_buffer(gst::Buffer buffer) const;

  /**
   * @brief Get the current negotiated caps from the sink pad
   *
   * Returns the actual caps negotiated during pipeline setup, not the allowed caps
   * set as a property. These are the fixed, fully-specified caps describing the
   * actual data format flowing through the pipeline.
   *
   * @return Negotiated caps with automatic reference counting, empty if not negotiated
   */
  gst::Caps get_current_caps() const;

  // Forward declaration for internal tensor builder (implementation detail)
  class TensorBuilder;

 private:
  // Static callback functions for appsink.
  static void eos_callback(::GstAppSink* appsink, ::gpointer user_data);
  static ::GstFlowReturn new_preroll_callback(::GstAppSink* appsink, ::gpointer user_data);
  static ::GstFlowReturn new_sample_callback(::GstAppSink* appsink, ::gpointer user_data);

  // Helper function for common buffer handling logic.
  ::GstFlowReturn handle_buffer_common(::GstAppSink* appsink,
                                       ::GstSample* (*pull_func)(::GstAppSink*),
                                       const char* callback_name);

  // Configuration.
  std::string name_;

  // GStreamer element (appsink)
  gst::AppSink sink_element_;

  // Queue of pending buffer pop requests (FIFO order).
  std::queue<std::promise<gst::Buffer>> pending_requests_;

  // Tensor builder for creating tensors from mapped memory.
  // Lazily initialized on first buffer based on caps (video vs generic format).
  // Owns a MemoryMapper for mapping GstMemory, and encapsulates all tensor
  // creation logic including name, shape, and strides.
  // Reused for all subsequent buffers for efficient memory management.
  mutable std::shared_ptr<TensorBuilder> tensor_builder_;

  // Synchronization.
  mutable std::mutex queue_mutex_;

  // Operation tracking for safe destruction.
  // Tracks EOS callbacks to prevent destruction while any operation is
  // accessing object members.
  GstWaitGroup active_operations_;

  // EOS synchronization.
  std::promise<void> eos_promise_;
  std::shared_future<void> eos_future_;
  std::atomic<bool> eos_signaled_{false};  // Prevents double-setting EOS promise
};

}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST_SINK_BRIDGE_HPP */
