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

#ifndef GST_SRC_RESOURCE_HPP
#define GST_SRC_RESOURCE_HPP

#include <chrono>
#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>

#include "gst/guards.hpp"
#include "gst/buffer.hpp"
#include "gst_src_bridge.hpp"

namespace holoscan {

/**
 * @brief Holoscan Resource wrapper for GStreamer appsrc element
 *
 * This class provides a Holoscan-specific interface to GStreamer pipelines,
 * integrating with Holoscan's entity system. It delegates actual GStreamer
 * operations to GstSrcBridge.
 */
class GstSrcResource : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(GstSrcResource)
  using SharedPtr = std::shared_ptr<GstSrcResource>;

  /**
   * @brief Destructor
   */
  ~GstSrcResource() = default;

  /**
   * @brief Setup the resource parameters
   */
  void setup(holoscan::ComponentSpec& spec) override;

  /**
   * @brief Initialize the GStreamer source resource
   */
  void initialize() override;

  /**
   * @brief Get the underlying GStreamer element
   * @return Shared future that will provide the GStreamer element when initialization completes
   */
  std::shared_future<GstElement*> get_gst_element() const {
    return element_future_;
  }

  /**
   * @brief Send End-Of-Stream signal to appsrc
   * 
   * Call this when you're done sending data to signal the downstream pipeline
   * to finalize processing (e.g., write file headers/trailers).
   * 
   * @param wait_ms Optional time to wait (in milliseconds) after sending EOS
   *                for GStreamer to process it. Default is 500ms.
   */
  void send_eos(int wait_ms = 500) {
    if (bridge_) { bridge_->send_eos(wait_ms); }
  }

  /**
   * @brief Push a buffer into the GStreamer pipeline (template version)
   * 
   * Template function that accepts any std::chrono::duration type and converts
   * it to milliseconds before calling the implementation.
   * 
   * @tparam Rep An arithmetic type representing the number of ticks
   * @tparam Period A std::ratio representing the tick period
   * @param buffer GStreamer buffer to push
   * @param timeout Duration to wait (zero = try immediately and return, no waiting)
   * @return true if buffer was successfully queued, false otherwise
   * 
   * @note For std::chrono::milliseconds, the non-template overload will be preferred
   *       by overload resolution.
   * 
   * @example
   *   resource->push_buffer(std::move(buffer), std::chrono::seconds(5));
   *   resource->push_buffer(std::move(buffer), std::chrono::minutes(1));
   *   resource->push_buffer(std::move(buffer), std::chrono::milliseconds(1000));  // calls non-template version
   */
  template<typename Rep, typename Period>
  bool push_buffer(holoscan::gst::Buffer buffer, std::chrono::duration<Rep, Period> timeout) {
    return push_buffer(std::move(buffer), std::chrono::duration_cast<std::chrono::milliseconds>(timeout));
  }

  /**
   * @brief Push a buffer into the GStreamer pipeline
   * 
   * Non-template overload for std::chrono::milliseconds. This is the actual implementation
   * and will be preferred by overload resolution when called with milliseconds.
   * 
   * If the queue is at capacity (controlled by queue_limit parameter), this function
   * will block until space becomes available, the timeout expires, or EOS is signaled.
   * 
   * @param buffer GStreamer buffer to push
   * @param timeout Duration to wait in milliseconds (zero = try immediately and return, no waiting)
   * @return true if buffer was successfully queued, false if buffer is invalid, 
   *         timeout expired, or EOS was signaled
   * 
   * @note This function provides backpressure: when the queue is full, the caller will
   *       block until GStreamer consumes a buffer or timeout expires, creating natural flow control.
   * @note If queue_limit is set to 0, the queue is unlimited and no blocking occurs.
   * 
   * @example
   *   // Try immediately, don't wait (default)
   *   resource->push_buffer(std::move(buffer));
   *   
   *   // Wait up to 1 second
   *   resource->push_buffer(std::move(buffer), std::chrono::milliseconds(1000));
   */
  bool push_buffer(holoscan::gst::Buffer buffer, 
                   std::chrono::milliseconds timeout = std::chrono::milliseconds::zero()) {
    return bridge_ ? bridge_->push_buffer(std::move(buffer), timeout) : false;
  }

  /**
   * @brief Create a GStreamer buffer from a GXF Entity containing tensor(s)
   * 
   * Supports both packed formats (RGBA, RGB) and planar formats (I420, NV12).
   * For multi-plane formats, expects separate tensors with naming convention:
   *   - "video_frame" for Y/luma plane
   *   - "video_frame_u", "video_frame_v" for chroma planes (I420)
   *   - "video_frame_uv" for interleaved chroma (NV12)
   * 
   * @param entity GXF Entity containing one or more tensors
   * @return GStreamer Buffer with zero-copy wrapping, empty on failure
   */
  holoscan::gst::Buffer create_buffer_from_entity(const holoscan::gxf::Entity& entity) const;

 private:
  // Bridge to GStreamer (does the actual work)
  std::shared_ptr<holoscan::gst::GstSrcBridge> bridge_;

  // Resource parameters
  holoscan::Parameter<std::string> caps_;
  holoscan::Parameter<size_t> queue_limit_;
  
  // Promise/future for element access (resolves after initialize())
  std::promise<GstElement*> element_promise_;
  std::shared_future<GstElement*> element_future_;
};

using GstSrcResourcePtr = GstSrcResource::SharedPtr;

}  // namespace holoscan

#endif /* GST_SRC_RESOURCE_HPP */

