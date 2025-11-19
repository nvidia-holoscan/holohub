/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN__GSTREAMER__GST_SRC_BRIDGE_HPP
#define HOLOSCAN__GSTREAMER__GST_SRC_BRIDGE_HPP

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>

#include <chrono>
#include <future>
#include <memory>
#include <string>

#include <holoscan/core/domain/tensor_map.hpp>

#include "gst/buffer.hpp"
#include "gst/caps.hpp"
#include "gst/config.hpp"
#include "gst/object.hpp"

namespace holoscan {

/**
 * @brief Bridge between GStreamer and external data sources
 *
 * This class provides a pure GStreamer implementation for managing an appsrc element,
 * handling buffer creation from tensors, and managing the data flow into GStreamer pipelines.
 * It contains no Holoscan-specific dependencies (except for GXF tensors which are the data format).
 *
 * @note The user is responsible for calling send_eos() when done sending data to properly
 *       finalize the stream. The destructor does NOT automatically send EOS.
 */
class GstSrcBridge {
 public:
  /**
   * @brief Constructor - creates and initializes the GStreamer appsrc element
   * @param name Optional name for the appsrc element
   * @param caps_string Capabilities string (e.g., "video/x-raw,format=RGBA,width=1920,height=1080")
   * @param max_buffers Maximum number of buffers in queue (0 = unlimited)
   * @param block If true, push_buffer() blocks when queue is full; if false, returns immediately
   * @throws std::runtime_error if initialization fails
   */
  GstSrcBridge(const std::string& name, const std::string& caps_string, size_t max_buffers, 
               bool block = true);

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
   * @return GStreamer element wrapper (appsrc)
   */
  gst::Element get_gst_element() const;

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
   * If the queue is at capacity (controlled by max_buffers), this function's
   * behavior depends on the block setting:
   * - If block=true (default): blocks until space becomes available or EOS is signaled
   * - If block=false: returns immediately, potentially dropping the buffer
   *
   * @param buffer GStreamer buffer to push
   * @return true if buffer was successfully queued, false otherwise
   */
  bool push_buffer(gst::Buffer buffer);

  /**
   * @brief Create a GStreamer buffer from a TensorMap
   *
   * Wraps all tensors in the map as GStreamer memory blocks with zero-copy.
   * Supports both packed formats (RGBA, RGB) and planar formats (I420, NV12).
   * For multi-plane formats, expects separate tensors with naming convention:
   *   - "video_frame" for Y/luma plane
   *   - "video_frame_u", "video_frame_v" for chroma planes (I420)
   *   - "video_frame_uv" for interleaved chroma (NV12)
   *
   * @param tensor_map TensorMap containing one or more tensors
   * @return GStreamer Buffer with zero-copy wrapping, empty on failure
   */
  gst::Buffer create_buffer_from_tensor_map(const TensorMap& tensor_map);

  /**
   * @brief Get the current negotiated caps from the source
   * @return Caps with automatic reference counting
   */
  gst::Caps get_caps() const;

  // Forward declarations for nested classes
  class MemoryWrapper;
  class HostMemoryWrapper;
#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
  class CudaMemoryWrapper;
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

 private:
  // Configuration
  std::string name_;
  std::string caps_string_;
  size_t max_buffers_;
  bool block_;  // Whether push_buffer() should block when queue is full

  // Framerate from caps (numerator/denominator)
  // Default to 0/1 (live mode - no framerate control)
  int framerate_num_ = 0;
  int framerate_den_ = 1;

  // GStreamer element (appsrc)
  gst::Element src_element_;

  // Memory wrapper factory for zero-copy tensor to GstMemory conversion.
  // Lazily initialized on first buffer creation based on tensor memory type (host/CUDA).
  // Handles wrapping tensor data pointers as GstMemory objects without copying data.
  // Reused for all subsequent buffers for efficient memory management.
  std::shared_ptr<MemoryWrapper> memory_wrapper_;

  // Frame timing
  uint64_t frame_count_ =
      0;  // Frame counter for accurate timestamp calculation (avoids rounding error accumulation)
};

}  // namespace holoscan

#endif /* HOLOSCAN__GSTREAMER__GST_SRC_BRIDGE_HPP */
