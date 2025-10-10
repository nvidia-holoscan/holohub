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

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include <gst/gst.h>
#include <gst/base/gstpushsrc.h>
#include <holoscan/holoscan.hpp>

#include "gst/guards.hpp"
#include "gst/buffer.hpp"
#include "gst/caps.hpp"

namespace holoscan {

/**
 * @brief Holoscan Resource wrapper for GStreamer source element
 *
 * This class provides a clean bridge between Holoscan operators and GStreamer pipelines.
 * The primary purpose is to enable Holoscan operators to push data into GStreamer pipelines
 * for further processing or output.
 */
class GstSrcResource : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(GstSrcResource)
  using SharedPtr = std::shared_ptr<GstSrcResource>;

  /**
   * @brief Destructor - cleans up GStreamer resources
   */
  ~GstSrcResource();

  /**
   * @brief Setup the resource parameters
   */
  void setup(holoscan::ComponentSpec& spec) override;

  /**
   * @brief Initialize the GStreamer source resource
   */
  void initialize() override;

  /**
   * @brief Check if the source element is ready (non-blocking)
   * @return true if the element has been initialized and is ready to use
   */
  bool valid() const {
    return src_element_future_.valid() && 
           src_element_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready && 
           src_element_future_.get();
  }

  /**
   * @brief Get the underlying GStreamer element (waits for initialization if needed)
   * @return Shared future that will provide the GstElementGuard when ready
   */
  std::shared_future<holoscan::gst::GstElementGuard> get_gst_element() const {
    return src_element_future_;
  }

  /**
   * @brief Push a buffer into the GStreamer pipeline
   * 
   * This function adds a buffer to the internal queue for consumption by GStreamer.
   * If the queue is at capacity (controlled by queue_limit parameter), this function
   * will block until space becomes available or until EOS is signaled.
   * 
   * @param buffer GStreamer buffer to push
   * @return true if buffer was successfully queued, false if buffer is invalid or EOS was signaled
   * 
   * @note This function provides backpressure: when the queue is full, the caller will
   *       block until GStreamer consumes a buffer, creating natural flow control.
   * @note If queue_limit is set to 0, the queue is unlimited and no blocking occurs.
   */
  bool push_buffer(holoscan::gst::Buffer buffer);

    /**
   * @brief Get the current negotiated caps from the source
   * @return Caps with automatic reference counting
   */
   holoscan::gst::Caps get_caps() const;

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

  // Static member functions for GStreamer callbacks
  static ::GstCaps* get_caps_callback(::GstBaseSrc *src, ::GstCaps *filter);
  static gboolean set_caps_callback(::GstBaseSrc *src, ::GstCaps *caps);
  static ::GstFlowReturn create_callback(::GstPushSrc *src, ::GstBuffer **buffer);
  static gboolean start_callback(::GstBaseSrc *src);
  static gboolean stop_callback(::GstBaseSrc *src);

 private:
  // Send EOS signal (called by destructor to unblock waiting threads)
  void send_eos();
  
  // Initialize memory wrapper based on tensor storage type and caps
  void initialize_memory_wrapper(nvidia::gxf::Tensor* tensor) const;

  // Promise/future for safe element access across threads
  std::promise<holoscan::gst::GstElementGuard> src_element_promise_;
  std::shared_future<holoscan::gst::GstElementGuard> src_element_future_;

  // Buffer queue for thread-safe async processing
  std::queue<holoscan::gst::Buffer> buffer_queue_;

  ::GstBuffer** pending_buffer_ = nullptr;
  mutable std::mutex mutex_;
  std::condition_variable queue_cv_;

  // EOS flag
  std::atomic<bool> eos_sent_{false};
  
  // Memory wrapper for tensor to GstMemory conversion (lazy initialization)
  // Forward declarations for nested classes
  class MemoryWrapper;
  class HostMemoryWrapper;
  class CudaMemoryWrapper;
  
  mutable std::shared_ptr<MemoryWrapper> memory_wrapper_;

  // Resource parameters
  holoscan::Parameter<std::string> caps_;
  holoscan::Parameter<size_t> queue_limit_;
};

using GstSrcResourcePtr = GstSrcResource::SharedPtr;

}  // namespace holoscan

#endif /* GST_SRC_RESOURCE_HPP */

