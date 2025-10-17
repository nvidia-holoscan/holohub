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
#include <cstring>
#include <memory>
#include <vector>

// ============================================================================
// Holoscan GstSrcResource Implementation
// Delegates to GstSrcBridge for all GStreamer operations
// ============================================================================

namespace holoscan {

holoscan::gst::Buffer GstSrcResource::create_buffer_from_entity(const gxf::Entity& entity) const {
  // Create an empty GStreamer buffer at the start
  holoscan::gst::Buffer gst_buffer;

  if (!entity) {
    HOLOSCAN_LOG_ERROR("Invalid entity provided");
    return gst_buffer;
  }

  if (!bridge_) {
    HOLOSCAN_LOG_ERROR("Bridge not initialized");
    return gst_buffer;
  }

  // Find all tensor components in the entity
  gxf_uid_t component_ids[64];  // Max 64 components
  uint64_t num_components = 64;
  gxf_result_t result = GxfComponentFindAll(entity.context(), entity.eid(), 
                                            &num_components, component_ids);
  if (result != GXF_SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to find components in entity");
    return gst_buffer;
  }

  // Collect all tensor pointers
  std::vector<nvidia::gxf::Tensor*> tensors;

  // Iterate through all components and collect tensors
  for (uint64_t i = 0; i < num_components; i++) {
    // Get component type info
    gxf_tid_t tid;
    result = GxfComponentType(entity.context(), component_ids[i], &tid);
    if (result != GXF_SUCCESS) {
      continue;
    }

    // Check if this is a Tensor component
    const char* type_name = nullptr;
    result = GxfComponentTypeName(entity.context(), tid, &type_name);
    if (result != GXF_SUCCESS || !type_name) {
      continue;
    }

    if (std::strcmp(type_name, "nvidia::gxf::Tensor") != 0) {
      continue;  // Not a tensor, skip
    }

    // Get tensor pointer
    void* tensor_ptr = nullptr;
    result = GxfComponentPointer(entity.context(), component_ids[i], 
                                  GxfTidNull(), &tensor_ptr);
    if (result != GXF_SUCCESS) {
      HOLOSCAN_LOG_WARN("Failed to get tensor pointer for component {}", i);
      continue;
    }

    auto* tensor = static_cast<nvidia::gxf::Tensor*>(tensor_ptr);
    tensors.push_back(tensor);
  }

  if (tensors.empty()) {
    HOLOSCAN_LOG_ERROR("No tensors found in entity");
  return gst_buffer;
  }

  // Delegate to bridge to create buffer from tensors
  return bridge_->create_buffer_from_tensors(tensors.data(), tensors.size());
}

void GstSrcResource::setup(holoscan::ComponentSpec& spec) {
  spec.param(caps_,
      "capabilities",
      "GStreamer Capabilities",
      "GStreamer caps string defining what data formats this source will provide. "
      "Use 'ANY' for maximum flexibility, or specify specific formats like "
      "'video/x-raw,format=RGBA,width=1920,height=1080' for video.",
      std::string("ANY"));
  spec.param(queue_limit_,
      "queue_limit",
      "Queue Limit",
      "Maximum number of buffers to keep in queue. When exceeded, push_buffer() will block. "
      "0 means unlimited queue size.",
      size_t(10));
}

void GstSrcResource::initialize() {
  // Call parent initialize first
  Resource::initialize();
  
  HOLOSCAN_LOG_INFO("Initializing GstSrcResource");
  HOLOSCAN_LOG_INFO("Configured capabilities: '{}'", caps_.get());
  HOLOSCAN_LOG_INFO("Queue limit: {}", queue_limit_.get());
  
  // Create the bridge (constructor initializes it)
  try {
    bridge_ = std::make_shared<holoscan::gst::GstSrcBridge>(
      name(), 
      caps_.get(), 
      queue_limit_.get()
    );
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to create GstSrcBridge: {}", e.what());
    throw;
  }
  
  HOLOSCAN_LOG_INFO("GstSrcResource initialized successfully");
}

}  // namespace holoscan

