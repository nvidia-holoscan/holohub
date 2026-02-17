/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gst_sink_resource.hpp"
#include "gst_sink_bridge.hpp"

#include <holoscan/core/execution_context.hpp>

// ============================================================================
// Holoscan GstSinkResource Implementation
// Delegates to GstSinkBridge for all GStreamer operations
// ============================================================================

namespace holoscan {

void GstSinkResource::setup(ComponentSpec& spec) {
  spec.param(caps_,
      "caps",
      "GStreamer Capabilities",
      "GStreamer caps string defining what data formats this sink can accept. "
      "Use 'ANY' for maximum flexibility, or specify specific formats like "
      "'video/x-raw,format=RGBA' for video or 'audio/x-raw' for audio.",
      std::string("ANY"));
  // Default max_buffers=1 is optimized for low-latency live display applications.
  // When Holoscan processing falls behind:
  //   - Small values (1): Frames are dropped, display stays "live" (shows current frames)
  //   - Large values (10+): Frames are queued, display shows delayed (but complete) data
  // For throughput-critical scenarios (batch processing, encoding), users should increase this.
  spec.param(max_buffers_,
      "max-buffers",
      "Max Buffers",
      "Maximum number of buffers to queue before blocking upstream (0 = unlimited, default: 1).",
      size_t(1));
  spec.param(qos_,
    "qos",
    "QoS",
    "Enable Quality of Service (QoS) in the sink. When enabled, frames may be dropped "
    "to maintain real-time performance. When disabled, all frames are processed.",
    false);
}

void GstSinkResource::initialize() {
  // Call parent initialize first
  Resource::initialize();

  HOLOSCAN_LOG_INFO("Initializing GstSinkResource");
  HOLOSCAN_LOG_INFO("Configured capabilities: '{}'", caps_.get());
  HOLOSCAN_LOG_INFO("Max buffers: {}", max_buffers_.get());
  HOLOSCAN_LOG_INFO("QoS enabled: {}", qos_.get());

  // Create the bridge (constructor initializes it)
  try {
    bridge_ = std::make_shared<GstSinkBridge>(
      name(),
      caps_.get(),
      max_buffers_.get(),
      qos_.get());

    // Set the promise with the GStreamer element so callers can wait for it
    element_promise_.set_value(bridge_->get_gst_element());
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to create GstSinkBridge: {}", e.what());
    element_promise_.set_exception(std::current_exception());
    throw;
  }

  HOLOSCAN_LOG_INFO("GstSinkResource initialized successfully");
}

std::shared_future<gst::Element> GstSinkResource::get_gst_element() const {
  return element_future_;
}

std::future<gst::Buffer> GstSinkResource::pull_buffer() {
  if (!bridge_) {
    HOLOSCAN_LOG_ERROR("Bridge not initialized");
    throw std::runtime_error("GstSinkResource not properly initialized");
  }
  return bridge_->pull_buffer();
}

TensorMap GstSinkResource::create_tensor_map_from_buffer(gst::Buffer buffer) const {
  if (!bridge_) {
    HOLOSCAN_LOG_ERROR("Bridge not initialized");
    return TensorMap();
  }

  // Delegate to bridge
  return bridge_->create_tensor_map_from_buffer(std::move(buffer));
}

}  // namespace holoscan
