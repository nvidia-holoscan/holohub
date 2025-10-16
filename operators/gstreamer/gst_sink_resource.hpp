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

#include <condition_variable>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <holoscan/holoscan.hpp>

#include "gst/guards.hpp"
#include "gst/buffer.hpp"
#include "gst/caps.hpp"

namespace holoscan {

/**
 * @brief Holoscan Resource wrapper for GStreamer appsink element
 *
 * This class provides a clean bridge between GStreamer pipelines and Holoscan operators
 * using the standard GStreamer appsink element. The primary purpose is to enable 
 * Holoscan operators to retrieve and process data from GStreamer pipelines.
 */
class GstSinkResource : public holoscan::Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(GstSinkResource)
  using SharedPtr = std::shared_ptr<GstSinkResource>;

  /**
   * @brief Destructor - cleans up GStreamer resources
   */
  ~GstSinkResource();

  /**
   * @brief Setup the resource parameters
   */
  void setup(holoscan::ComponentSpec& spec) override;

  /**
   * @brief Initialize the GStreamer sink resource
   */
  void initialize() override;

  /**
   * @brief Get the underlying GStreamer element (waits for initialization if needed)
   * @return Shared future that will provide the GstElementGuard when ready
   */
  std::shared_future<holoscan::gst::GstElementGuard> get_gst_element() const {
    return sink_element_future_;
  }

  /**
   * @brief Asynchronously pop the next buffer from the GStreamer pipeline
   * @return Future that will be fulfilled when a buffer becomes available
   */
  std::future<holoscan::gst::Buffer> pop_buffer();

  /**
   * @brief Create a GXF Entity with tensor(s) from GStreamer buffer with zero-copy
   * 
   * Supports both packed formats (RGBA, RGB) and planar formats (I420, NV12).
   * For multi-plane formats, creates separate tensors with naming convention:
   *   - "video_frame" for Y/luma plane
   *   - "video_frame_u", "video_frame_v" for chroma planes (I420)
   *   - "video_frame_uv" for interleaved chroma (NV12)
   * 
   * @param context Execution context for GXF operations
   * @param buffer GStreamer buffer containing the data
   * @return GXF Entity containing one or more tensors, empty entity on failure
   */
  holoscan::gxf::Entity create_entity_from_buffer(holoscan::ExecutionContext& context,
                                                   const holoscan::gst::Buffer& buffer) const;

 private:
  // Friend declarations for appsink callback functions (static functions in holoscan namespace)
  friend void appsink_eos_callback(::GstAppSink* appsink, gpointer user_data);
  friend ::GstFlowReturn appsink_new_preroll_callback(::GstAppSink* appsink, gpointer user_data);
  friend ::GstFlowReturn appsink_new_sample_callback(::GstAppSink* appsink, gpointer user_data);
  /**
   * @brief Check if the sink element is ready (non-blocking)
   * @return true if the element has been initialized and is ready to use
   */
   bool valid() const {
    return sink_element_future_.valid() && 
           sink_element_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready && 
           sink_element_future_.get();
  }

  /**
   * @brief Get the current negotiated caps from the sink
   * @return Caps with automatic reference counting
   */
   holoscan::gst::Caps get_caps() const;

 private:
  /**
   * @brief Tensor metadata (name, shape, strides)
   */
  struct TensorMetadata {
    std::string name;
    nvidia::gxf::Shape shape;
    std::array<size_t, 8> strides;
  };

  /**
   * @brief Get tensor metadata for video plane
   * 
   * @param video_info Video format information
   * @param plane_idx Index of the plane
   * @param n_planes Total number of planes
   * @return Tensor metadata (name, shape, strides)
   */
  TensorMetadata get_video_tensor_metadata(
      const holoscan::gst::VideoInfo& video_info,
      guint plane_idx,
      guint n_planes) const;

  /**
   * @brief Get tensor metadata for generic memory block
   * 
   * @param mem_idx Index of the memory block
   * @param size Size of the memory block in bytes
   * @param n_mem Total number of memory blocks
   * @return Tensor metadata (name, shape, strides)
   */
  TensorMetadata get_generic_tensor_metadata(
      guint mem_idx,
      gsize size,
      guint n_mem) const;

  // Promise/future for safe element access across threads
  std::promise<holoscan::gst::GstElementGuard> sink_element_promise_;
  std::shared_future<holoscan::gst::GstElementGuard> sink_element_future_;

  // Buffer queue for thread-safe async processing
  std::queue<holoscan::gst::Buffer> buffer_queue_;
  // Single pending buffer pop request (only one request at a time)
  std::optional<std::promise<holoscan::gst::Buffer>> pending_request_;
  mutable std::mutex mutex_;
  std::condition_variable queue_cv_;
  bool is_shutting_down_ = false;

  // Resource parameters
  holoscan::Parameter<std::string> caps_;
  holoscan::Parameter<bool> qos_enabled_;
  holoscan::Parameter<size_t> queue_limit_;
};

using GstSinkResourcePtr = GstSinkResource::SharedPtr;

}  // namespace holoscan

#endif /* GST_SINK_RESOURCE_HPP */
