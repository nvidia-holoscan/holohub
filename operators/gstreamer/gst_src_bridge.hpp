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

#ifndef GST_SRC_BRIDGE_HPP
#define GST_SRC_BRIDGE_HPP

#include <chrono>
#include <future>
#include <memory>
#include <string>

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gxf/std/tensor.hpp>

#include "gst/object.hpp"
#include "gst/guards.hpp"
#include "gst/buffer.hpp"
#include "gst/caps.hpp"

namespace holoscan {
namespace gst {

/**
 * @brief Bridge between GStreamer and external data sources
 *
 * This class provides a pure GStreamer implementation for managing an appsrc element,
 * handling buffer creation from tensors, and managing the data flow into GStreamer pipelines.
 * It contains no Holoscan-specific dependencies (except for GXF tensors which are the data format).
 */
class GstSrcBridge {
 public:
  /**
   * @brief Constructor - creates and initializes the GStreamer appsrc element
   * @param name Optional name for the appsrc element
   * @param caps Capabilities string (e.g., "video/x-raw,format=RGBA,width=1920,height=1080")
   * @param queue_limit Maximum number of buffers in queue (0 = unlimited)
   * @throws std::runtime_error if initialization fails
   */
  GstSrcBridge(const std::string& name, const std::string& caps, size_t queue_limit);

  /**
   * @brief Destructor - cleans up GStreamer resources
   */
  ~GstSrcBridge();

  // Non-copyable and non-movable
  GstSrcBridge(const GstSrcBridge&) = delete;
  GstSrcBridge& operator=(const GstSrcBridge&) = delete;
  GstSrcBridge(GstSrcBridge&&) = delete;
  GstSrcBridge& operator=(GstSrcBridge&&) = delete;

  /**
   * @brief Get the underlying GStreamer element
   * @return Reference to the GStreamer element wrapper (appsrc)
   */
  const Element& get_gst_element() const {
    return src_element_;
  }

  /**
   * @brief Send End-Of-Stream signal to appsrc
   * 
   * Call this when you're done sending data to signal the downstream pipeline
   * to finalize processing (e.g., write file headers/trailers).
   * 
   * Note: This function returns immediately after sending EOS. The caller should
   * wait for the EOS message on the pipeline bus to know when processing is complete.
   * 
   * @return true if EOS was successfully sent, false if already sent or error occurred
   */
  bool send_eos();

  /**
   * @brief Push a buffer into the GStreamer pipeline
   * 
   * If the queue is at capacity (controlled by queue_limit), this function
   * will block until space becomes available, the timeout expires, or EOS is signaled.
   * 
   * @param buffer GStreamer buffer to push
   * @param timeout Duration to wait in milliseconds (zero = try immediately and return, no waiting)
   * @return true if buffer was successfully queued, false otherwise
   */
  bool push_buffer(Buffer buffer, std::chrono::milliseconds timeout = std::chrono::milliseconds::zero());

  /**
   * @brief Create a GStreamer buffer from raw tensor data
   * 
   * Wraps tensors in GStreamer memory using zero-copy when possible.
   * Automatically selects between host and CUDA memory based on tensor storage type and caps.
   * 
   * @param tensors Array of tensor pointers
   * @param num_tensors Number of tensors in the array
   * @return GStreamer Buffer with zero-copy wrapping, empty on failure
   */
  Buffer create_buffer_from_tensors(nvidia::gxf::Tensor** tensors, size_t num_tensors);

  /**
   * @brief Create a GStreamer buffer from a GXF Entity containing tensor(s)
   * 
   * Extracts all tensors from the entity and wraps them in a GStreamer buffer.
   * Supports both packed formats (RGBA, RGB) and planar formats (I420, NV12).
   * For multi-plane formats, expects separate tensors with naming convention:
   *   - "video_frame" for Y/luma plane
   *   - "video_frame_u", "video_frame_v" for chroma planes (I420)
   *   - "video_frame_uv" for interleaved chroma (NV12)
   * 
   * @param entity GXF Entity containing one or more tensors
   * @return GStreamer Buffer with zero-copy wrapping, empty on failure
   */
  Buffer create_buffer_from_entity(const nvidia::gxf::Entity& entity);

  /**
   * @brief Get the current negotiated caps from the source
   * @return Caps with automatic reference counting
   */
  Caps get_caps() const;

 private:
  // Forward declarations for nested classes
  class MemoryWrapper;
  class HostMemoryWrapper;
  class CudaMemoryWrapper;

  /**
   * @brief Initialize memory wrapper based on tensor storage type and caps
   */
  void initialize_memory_wrapper(nvidia::gxf::Tensor* tensor);

  // Configuration
  std::string name_;
  std::string caps_;
  size_t queue_limit_;

  // Framerate from caps (numerator/denominator)
  // Default to 0/1 (live mode - no framerate control)
  int framerate_num_ = 0;
  int framerate_den_ = 1;

  // GStreamer element (appsrc)
  Element src_element_;

  // Memory wrapper for tensor to GstMemory conversion (lazy initialization)
  std::shared_ptr<MemoryWrapper> memory_wrapper_;
  
  // Frame timing
  uint64_t frame_count_ = 0;  // Frame counter for accurate timestamp calculation (avoids rounding error accumulation)
};

}  // namespace gst
}  // namespace holoscan

#endif /* GST_SRC_BRIDGE_HPP */

