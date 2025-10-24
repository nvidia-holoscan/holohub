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
  spec.param(max_buffers_,
      "max-buffers",
      "Max Buffers",
      "Maximum number of buffers to keep in queue. The render callback will block when "
      "queue size exceeds this limit. 0 means one buffer at a time (blocks until consumed).",
      size_t(1));
  spec.param(qos_enabled_,
    "qos-enabled",
    "QoS Enabled",
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
  HOLOSCAN_LOG_INFO("QoS enabled: {}", qos_enabled_.get());
  
  // Create the bridge (constructor initializes it)
  try {
    bridge_ = std::make_shared<GstSinkBridge>(
      name(), 
      caps_.get(), 
      max_buffers_.get(),
      qos_enabled_.get()
    );
    
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

std::future<gst::Buffer> GstSinkResource::pop_buffer() {
  return bridge_ ? bridge_->pop_buffer() : std::future<gst::Buffer>();
}

gxf::Entity GstSinkResource::create_entity_from_buffer(
    ExecutionContext& context,
    const gst::Buffer& buffer) const {
  if (!bridge_) {
    HOLOSCAN_LOG_ERROR("Bridge not initialized");
    return gxf::Entity();
  }
  
  // Delegate to bridge
  return bridge_->create_entity_from_buffer(context, buffer);
}

}  // namespace holoscan
