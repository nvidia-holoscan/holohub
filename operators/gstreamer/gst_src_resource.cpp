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

#include "gst_src_resource.hpp"
#include "gst_src_bridge.hpp"

#include <holoscan/core/execution_context.hpp>

// ============================================================================
// Holoscan GstSrcResource Implementation
// Delegates to GstSrcBridge for all GStreamer operations
// ============================================================================

namespace holoscan {

void GstSrcResource::setup(ComponentSpec& spec) {
  spec.param(caps_,
      "caps",
      "GStreamer Capabilities",
      "GStreamer caps string defining what data formats this source will provide. "
      "Use 'ANY' for maximum flexibility, or specify specific formats like "
      "'video/x-raw,format=RGBA,width=1920,height=1080' for video.",
      std::string("ANY"));
  spec.param(max_buffers_,
      "max-buffers",
      "Max Buffers",
      "Maximum number of buffers to keep in queue. When exceeded, push_buffer() will block. "
      "0 means unlimited queue size.",
      size_t(10));
}

void GstSrcResource::initialize() {
  // Call parent initialize first
  Resource::initialize();
  
  HOLOSCAN_LOG_INFO("Initializing GstSrcResource");
  HOLOSCAN_LOG_INFO("Configured capabilities: '{}'", caps_.get());
  HOLOSCAN_LOG_INFO("Max buffers: {}", max_buffers_.get());
  
  // Create the bridge (constructor initializes it)
  try {
    bridge_ = std::make_shared<GstSrcBridge>(
      name(), 
      caps_.get(), 
      max_buffers_.get()
    );
    
    // Set the promise with the GStreamer element so callers can wait for it
    element_promise_.set_value(bridge_->get_gst_element());
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to create GstSrcBridge: {}", e.what());
    element_promise_.set_exception(std::current_exception());
    throw;
  }
  
  HOLOSCAN_LOG_INFO("GstSrcResource initialized successfully");
}

std::shared_future<gst::Element> GstSrcResource::get_gst_element() const {
  return element_future_;
}

bool GstSrcResource::push_buffer(gst::Buffer buffer, std::chrono::milliseconds timeout) {
  return bridge_ ? bridge_->push_buffer(std::move(buffer), timeout) : false;
}

void GstSrcResource::send_eos() {
  if (bridge_) {
    bridge_->send_eos();
  }
}

gst::Buffer GstSrcResource::create_buffer_from_entity(const gxf::Entity& entity) const {
  if (!bridge_) {
    HOLOSCAN_LOG_ERROR("Bridge not initialized");
    return gst::Buffer();
  }
  
  // Delegate to bridge
  return bridge_->create_buffer_from_entity(entity);
}

}  // namespace holoscan

