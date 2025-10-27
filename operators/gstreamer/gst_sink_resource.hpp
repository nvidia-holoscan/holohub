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
#include <string>

#include <holoscan/holoscan.hpp>

#include "gst/buffer.hpp"
#include "gst_sink_bridge.hpp"

namespace holoscan {

/**
 * @brief Holoscan Resource wrapper for GStreamer appsink element
 *
 * This class provides a Holoscan-specific interface to GStreamer pipelines,
 * integrating with Holoscan's entity system. It delegates actual GStreamer
 * operations to GstSinkBridge.
 */
class GstSinkResource : public Resource {
 public:
  HOLOSCAN_RESOURCE_FORWARD_ARGS(GstSinkResource)

  /**
   * @brief Setup the resource parameters
   */
  void setup(ComponentSpec& spec) override;

  /**
   * @brief Initialize the GStreamer sink resource
   */
  void initialize() override;

  /**
   * @brief Get the underlying GStreamer element
   * @return Shared future that will provide the GStreamer element when initialization completes
   */
  std::shared_future<gst::Element> get_gst_element() const;

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

 private:
  // Bridge to GStreamer (does the actual work)
  std::shared_ptr<GstSinkBridge> bridge_;

  // Resource parameters
  Parameter<std::string> caps_;
  Parameter<size_t> max_buffers_;
  Parameter<bool> qos_enabled_;
  
  // Promise/future for element access (resolves after initialize())
  std::promise<gst::Element> element_promise_;
  std::shared_future<gst::Element> element_future_ = element_promise_.get_future();
};

}  // namespace holoscan

#endif /* GST_SINK_RESOURCE_HPP */
