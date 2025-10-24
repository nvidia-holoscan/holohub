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
#include "gst/guards.hpp"
#include "gst/buffer.hpp"
#include "gst/caps.hpp"
#include "gst/video_info.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/base/gstbasesink.h>
#include <gst/video/video.h>
#include <gst/cuda/gstcudamemory.h>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <holoscan/core/execution_context.hpp>
#include <gxf/std/tensor.hpp>

// Convenience constant for mapping CUDA memory for reading
// GST_MAP_CUDA is provided by gst/cuda/gstcudamemory.h
// GST_MAP_READ_CUDA is available in GStreamer >= 1.28, define it for older versions
#ifndef GST_MAP_READ_CUDA
#define GST_MAP_READ_CUDA ((GstMapFlags) (GST_MAP_READ | GST_MAP_CUDA))
#endif

namespace {

/**
 * @brief Helper function to map GStreamer memory with the specified flags
 * @param memory GStreamer memory block to map
 * @param map_info Output map info structure
 * @param flags Mapping flags (GST_MAP_READ, GST_MAP_READ_CUDA, etc.)
 * @param mapped_device_type Output device type detected by CUDA API
 * @param storage_type Output GXF storage type based on device type
 * @param data_ptr Output pointer to mapped data
 * @param size Output size of mapped memory
 * @return true if mapping succeeded, false otherwise
 */
bool map_gst_memory(::GstMemory* memory, ::GstMapInfo& map_info, ::GstMapFlags flags,
                    DLDeviceType& mapped_device_type, nvidia::gxf::MemoryStorageType& storage_type,
                    void*& data_ptr, gsize& size) {
  if (!gst_memory_map(memory, &map_info, flags)) {
    return false;
  }
  
  // Use CUDA API to accurately detect the memory type
  cudaPointerAttributes attributes;
  cudaError_t result = cudaPointerGetAttributes(&attributes, map_info.data);
  
  if (result != cudaSuccess) {
    cudaGetLastError();  // Reset error
    mapped_device_type = kDLCPU;
  } else {
    // Convert CUDA memory type to DLDeviceType
    switch (attributes.type) {
      case cudaMemoryTypeDevice:
        mapped_device_type = kDLCUDA;
        break;
      case cudaMemoryTypeHost:
        mapped_device_type = kDLCUDAHost;
        break;
      case cudaMemoryTypeManaged:
        mapped_device_type = kDLCUDAManaged;
        break;
      default:
        mapped_device_type = kDLCPU;
    }
  }
  
  // Set storage type based on actual device type
  if (mapped_device_type == kDLCUDA || mapped_device_type == kDLCUDAManaged) {
    storage_type = nvidia::gxf::MemoryStorageType::kDevice;
  } else if (mapped_device_type == kDLCUDAHost) {
    storage_type = nvidia::gxf::MemoryStorageType::kHost;
  } else {
    storage_type = nvidia::gxf::MemoryStorageType::kSystem;
  }
  
  data_ptr = map_info.data;
  size = map_info.size;
  
  return true;
}

}  // namespace

// ============================================================================
// Holoscan GstSinkResource Implementation (C++)
// Using standard GStreamer appsink element instead of custom element
// ============================================================================

namespace holoscan {

// Forward declare the callback functions (defined below, after the unnamed namespace)
// Note: These cannot be static because they are friend functions
void appsink_eos_callback(GstAppSink* appsink, gpointer user_data);
GstFlowReturn appsink_new_preroll_callback(GstAppSink* appsink, gpointer user_data);
GstFlowReturn appsink_new_sample_callback(GstAppSink* appsink, gpointer user_data);

namespace {
// Factory function implementations moved to common.cpp

// Helper function to extract media type from caps (internal use only)
const char* get_media_type_from_caps(::GstCaps* caps) {
  if (!caps || gst_caps_is_empty(caps) || gst_caps_get_size(caps) == 0) {
    return nullptr;
  }

  ::GstStructure* structure = gst_caps_get_structure(caps, 0);
  if (!structure) {
    return nullptr;
  }

  return gst_structure_get_name(structure);
}

} // unnamed namespace

// Callback functions for appsink - must be outside anonymous namespace
// These are friend functions of GstSinkResource for access to private members

void appsink_eos_callback(GstAppSink* appsink, gpointer user_data) {
  GstSinkResource* resource = static_cast<GstSinkResource*>(user_data);
  HOLOSCAN_LOG_INFO("End of stream in appsink: {}", resource->name());
}

GstFlowReturn appsink_new_preroll_callback(GstAppSink* appsink, gpointer user_data) {
  GstSinkResource* resource = static_cast<GstSinkResource*>(user_data);
  
  // Check if shutting down before pulling sample
  {
    std::unique_lock<std::mutex> lock(resource->mutex_);
    if (resource->is_shutting_down_) {
      return GST_FLOW_EOS;
    }
  }
  
  GstSample* sample = gst_app_sink_pull_preroll(appsink);
  if (!sample) {
    return GST_FLOW_ERROR;
  }
  
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    gst_sample_unref(sample);
    return GST_FLOW_ERROR;
  }
  
  HOLOSCAN_LOG_DEBUG("Received preroll buffer via appsink, size: {} bytes", 
                     gst_buffer_get_size(buffer));
  
  std::unique_lock<std::mutex> lock(resource->mutex_);
  
  // Double-check shutdown flag after acquiring lock
  if (resource->is_shutting_down_) {
    gst_sample_unref(sample);
    return GST_FLOW_EOS;
  }
  
  gst::Buffer buffer_obj(gst_buffer_ref(buffer));
  
  if (resource->pending_request_.has_value()) {
    std::promise<gst::Buffer> promise = std::move(resource->pending_request_.value());
    resource->pending_request_.reset();
    promise.set_value(std::move(buffer_obj));
    HOLOSCAN_LOG_DEBUG("Fulfilled pending buffer request with preroll");
  } else {
    resource->buffer_queue_.push(std::move(buffer_obj));
    HOLOSCAN_LOG_DEBUG("Queued preroll buffer, total in queue: {}", 
                       resource->buffer_queue_.size());
  }
  
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

GstFlowReturn appsink_new_sample_callback(GstAppSink* appsink, gpointer user_data) {
  GstSinkResource* resource = static_cast<GstSinkResource*>(user_data);
  
  // Check if shutting down before pulling sample
  {
    std::unique_lock<std::mutex> lock(resource->mutex_);
    if (resource->is_shutting_down_) {
      return GST_FLOW_EOS;
    }
  }
  
  // Pull the sample (contains buffer + caps)
  GstSample* sample = gst_app_sink_pull_sample(appsink);
  if (!sample) {
    return GST_FLOW_ERROR;
  }
  
  // Extract buffer from sample
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    gst_sample_unref(sample);
    return GST_FLOW_ERROR;
  }
  
  HOLOSCAN_LOG_DEBUG("Received buffer via appsink, size: {} bytes", 
                     gst_buffer_get_size(buffer));
  
  // Lock for thread-safe queue access
  std::unique_lock<std::mutex> lock(resource->mutex_);
  
  // Double-check shutdown flag after acquiring lock
  if (resource->is_shutting_down_) {
    gst_sample_unref(sample);
    return GST_FLOW_EOS;
  }
  
  // Create Buffer object (ref the buffer before unreffing sample)
  gst::Buffer buffer_obj(gst_buffer_ref(buffer));
  
  // Check if there is a pending request waiting for a buffer
  if (resource->pending_request_.has_value()) {
    // Fulfill the pending request immediately
    std::promise<gst::Buffer> promise = std::move(resource->pending_request_.value());
    resource->pending_request_.reset();
    promise.set_value(std::move(buffer_obj));
    HOLOSCAN_LOG_DEBUG("Fulfilled pending buffer request");
  } else {
    // Queue the buffer for future requests
    resource->buffer_queue_.push(std::move(buffer_obj));
    HOLOSCAN_LOG_DEBUG("Queued buffer, total in queue: {}", resource->buffer_queue_.size());
    
    // Backpressure: wait if queue is full
    size_t max_buffers = resource->max_buffers_.get();
    if (resource->buffer_queue_.size() > max_buffers) {
      HOLOSCAN_LOG_DEBUG("Buffer queue size ({}) exceeds limit ({}), waiting...", 
                        resource->buffer_queue_.size(), max_buffers);
      
      resource->queue_cv_.wait(lock, [resource, max_buffers]() {
        return resource->is_shutting_down_ || resource->buffer_queue_.size() <= max_buffers;
      });
      
      // Check if we woke up due to shutdown
      if (resource->is_shutting_down_) {
        gst_sample_unref(sample);
        return GST_FLOW_EOS;
      }
      
      HOLOSCAN_LOG_DEBUG("Buffer queue size now within limit: {}", 
                        resource->buffer_queue_.size());
    }
  }
  
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

// Asynchronously pop next buffer using promise-based approach
std::future<gst::Buffer> GstSinkResource::pop_buffer() {
  std::lock_guard<std::mutex> lock(mutex_);

  // Create a promise for this request
  std::promise<gst::Buffer> promise;
  auto future = promise.get_future();

  // Check if we have buffers available immediately
  if (!buffer_queue_.empty()) {
    // Fulfill promise immediately with available buffer
    gst::Buffer buffer = std::move(buffer_queue_.front());
    buffer_queue_.pop();
    promise.set_value(std::move(buffer));
    HOLOSCAN_LOG_DEBUG("Fulfilled buffer request immediately, remaining buffers: {}", buffer_queue_.size());
    
    // Notify waiting producers that space is available in the queue
    queue_cv_.notify_one();
  } else {
    // No buffers available, store the promise for later fulfillment
    if (pending_request_.has_value()) {
      HOLOSCAN_LOG_WARN("Multiple concurrent pop_buffer() requests detected. "
                        "Only one request is supported at a time. Previous request will be overwritten.");
    }
    pending_request_ = std::move(promise);
    HOLOSCAN_LOG_DEBUG("Stored pending buffer request");
  }

  return future;
}

// Get current negotiated caps
gst::Caps GstSinkResource::get_caps() const {
  // Check if element is ready and valid
  if (!valid()) {
    return gst::Caps(); // Return empty caps if not ready
  }

  // Get the sink pad and its current caps
  ::GstPad* pad = gst_element_get_static_pad(sink_element_future_.get().get(), "sink");
  if (!pad) {
    return gst::Caps(); // Return empty caps
  }

  ::GstCaps* caps = gst_pad_get_current_caps(pad);
  gst_object_unref(pad);

  return gst::Caps(caps); // Automatic reference counting
}

GstSinkResource::TensorMetadata GstSinkResource::get_video_tensor_metadata(
    const gst::VideoInfo& video_info,
    guint plane_idx,
    guint n_planes) const {
  
  // Get plane-specific dimensions
  int plane_width = GST_VIDEO_INFO_COMP_WIDTH(video_info.get(), plane_idx);
  int plane_height = GST_VIDEO_INFO_COMP_HEIGHT(video_info.get(), plane_idx);
  int plane_stride = GST_VIDEO_INFO_PLANE_STRIDE(video_info.get(), plane_idx);
  guint plane_components = (plane_idx == 0 && n_planes == 1) ? 
                           GST_VIDEO_INFO_N_COMPONENTS(video_info.get()) : 1;
  
  // Create tensor name with appropriate suffix
  static const char* plane_suffixes[] = {"", "_u", "_v", "_a"};
  std::string tensor_name = "video_frame";
  if (n_planes > 1) {
    if (n_planes == 2 && plane_idx == 1) {
      tensor_name += "_uv";  // NV12 format
    } else if (plane_idx > 0) {
      tensor_name += plane_suffixes[plane_idx];  // I420 format
    }
  }
  
  // Create shape: [height, width, components]
  nvidia::gxf::Shape shape({static_cast<int32_t>(plane_height), 
                            static_cast<int32_t>(plane_width),
                            static_cast<int32_t>(plane_components)});
  
  // Calculate strides: [row_stride, bytes_per_pixel, bytes_per_component]
  size_t bytes_per_pixel = plane_components;
  std::array<size_t, 8> strides{{
    static_cast<size_t>(plane_stride),
    bytes_per_pixel,
    1,  // uint8
    0, 0, 0, 0, 0
  }};
  
  return {tensor_name, shape, strides};
}

GstSinkResource::TensorMetadata GstSinkResource::get_generic_tensor_metadata(
    guint mem_idx,
    gsize size,
    guint n_mem) const {
  
  // Create tensor name
  std::string tensor_name = (n_mem == 1) ? "data" : fmt::format("data_{}", mem_idx);
  
  // Create shape: 1D byte array
  nvidia::gxf::Shape shape({static_cast<int32_t>(size)});
  
  // Strides: just element size
  std::array<size_t, 8> strides{{1, 0, 0, 0, 0, 0, 0, 0}};
  
  return {tensor_name, shape, strides};
}

gxf::Entity GstSinkResource::create_entity_from_buffer(
    ExecutionContext& context,
    const gst::Buffer& buffer) const {
  
  // Get current caps
  gst::Caps caps = get_caps();
  
  // Create entity to hold tensor(s)
  auto entity = gxf::Entity::New(&context);

  // Validate caps
  if (caps.is_empty()) {
    HOLOSCAN_LOG_ERROR("No caps available for buffer");
    return gxf::Entity();
  }

  // Check if this is video/x-raw data
  auto video_info_opt = caps.get_video_info();
  
  // Get number of memory blocks in the buffer
  guint n_mem = gst_buffer_n_memory(buffer.get());
  
  
  // Process each memory block
  for (guint mem_idx = 0; mem_idx < n_mem; mem_idx++) {
    ::GstMemory* memory = gst_buffer_peek_memory(buffer.get(), mem_idx);
    if (!memory) {
      HOLOSCAN_LOG_ERROR("No memory found for block {}", mem_idx);
      return gxf::Entity();
    }
    
    // Get tensor metadata based on data type
    TensorMetadata metadata;
    if (video_info_opt.has_value()) {
      const gst::VideoInfo& video_info = *video_info_opt;
      guint n_planes = video_info->finfo->n_planes;
      metadata = get_video_tensor_metadata(video_info, mem_idx, n_planes);
    } else {
      gsize size = gst_memory_get_sizes(memory, nullptr, nullptr);
      metadata = get_generic_tensor_metadata(mem_idx, size, n_mem);
    }
    
    // Map memory (try CUDA first if requested, fallback to CPU)
    ::GstMapInfo map_info;
    void* data_ptr = nullptr;
    gsize size = 0;
    DLDeviceType mapped_device_type = kDLCPU;
    nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kSystem;
    
    bool memory_mapped = false;
    // Check if caps indicate CUDA memory, if so, try CUDA mapping first if it fails, fallback to CPU mapping
    if (caps.has_feature(GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY)) {
      if (map_gst_memory(memory, map_info, GST_MAP_READ_CUDA, mapped_device_type, 
                        storage_type, data_ptr, size)) {
        memory_mapped = true;
      }
    }
    if (!memory_mapped) {
      if (!map_gst_memory(memory, map_info, GST_MAP_READ, mapped_device_type, 
                         storage_type, data_ptr, size)) {
        HOLOSCAN_LOG_ERROR("Failed to map memory for block {}", mem_idx);
        return gxf::Entity();
      }
    }
    
    // Add tensor to entity
    auto gxf_tensor_result = static_cast<nvidia::gxf::Entity&>(entity)
                               .add<nvidia::gxf::Tensor>(metadata.name.c_str());
    if (!gxf_tensor_result) {
      HOLOSCAN_LOG_ERROR("Failed to add tensor '{}'", metadata.name);
      gst_memory_unmap(memory, &map_info);
      return gxf::Entity();
    }
    auto gxf_tensor = gxf_tensor_result.value();
    
    // Wrap memory as tensor with deleter for cleanup
    nvidia::gxf::PrimitiveType primitive_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    uint64_t element_size = nvidia::gxf::PrimitiveTypeSize(primitive_type);
    
    std::function<nvidia::gxf::Expected<void>(void*)> deleter = 
        [buffer, memory, map_info](void*) mutable {
          gst_memory_unmap(memory, const_cast<::GstMapInfo*>(&map_info));
          return nvidia::gxf::Success;
        };
    
    gxf_tensor->wrapMemory(
      metadata.shape,
      primitive_type,
      element_size,
      metadata.strides,
      storage_type,
      static_cast<uint8_t*>(data_ptr),
      deleter);
  }
  
  return entity;
}

GstSinkResource::~GstSinkResource() {
  HOLOSCAN_LOG_INFO("Destroying GstSinkResource");
  // Signal shutdown and wake up any waiting callbacks
  {
    std::lock_guard<std::mutex> lock(mutex_);
    is_shutting_down_ = true;
    std::queue<gst::Buffer> empty_queue;
    std::swap(buffer_queue_, empty_queue);
    pending_request_.reset();
    queue_cv_.notify_all();
  }
  HOLOSCAN_LOG_INFO("GstSinkResource destroyed");
}

void GstSinkResource::setup(holoscan::ComponentSpec& spec) {
  spec.param(caps_,
      "caps",
      "GStreamer Capabilities",
      "GStreamer caps string defining what data formats this sink can accept. "
      "Use 'ANY' for maximum flexibility, or specify specific formats like "
      "'video/x-raw,format=RGBA' for video or 'audio/x-raw' for audio.",
      std::string("ANY"));
  spec.param(qos_enabled_,
      "qos-enabled",
      "QoS Enabled",
      "Enable Quality of Service (QoS) in the sink. When enabled, frames may be dropped "
      "to maintain real-time performance. When disabled, all frames are processed.",
      false);
  spec.param(max_buffers_,
      "max-buffers",
      "Max Buffers",
      "Maximum number of buffers to keep in queue. The render callback will block when "
      "queue size exceeds this limit. 0 means one buffer at a time (blocks until consumed).",
      size_t(1));
}

void GstSinkResource::initialize() {
  // Call parent initialize first
  Resource::initialize();
  
  // Initialize the future from the promise (after any construction/moves are complete)
  sink_element_future_ = sink_element_promise_.get_future();
  
  // Initialize shutdown flag
  is_shutting_down_ = false;
  
  HOLOSCAN_LOG_INFO("Initializing GstSinkResource with appsink for data bridging");
  HOLOSCAN_LOG_INFO("Configured capabilities: '{}'", caps_.get());
  
  // Initialize GStreamer if not already done
  if (!gst_is_initialized()) {
    gst_init(nullptr, nullptr);
  }

  // Create appsink element - standard GStreamer element, no custom registration needed!
  auto element = gst::Element(
    gst_element_factory_make("appsink", name().empty() ? nullptr : name().c_str())
  );

  if (!element) {
    HOLOSCAN_LOG_ERROR("Failed to create appsink element");
    sink_element_promise_.set_exception(
      std::make_exception_ptr(std::runtime_error("Failed to create appsink element"))
    );
    return;
  }

  GstAppSink* appsink = GST_APP_SINK(element.get());

  // Configure appsink properties
  g_object_set(appsink,
    "emit-signals", FALSE,  // Use callbacks instead of signals (more efficient)
    "sync", TRUE,           // Sync to clock for proper timing
    "max-buffers", static_cast<guint>(max_buffers_.get() + 1),  // Buffer queue limit
    "drop", FALSE,          // Don't drop buffers - we handle backpressure
    NULL
  );

  // Set caps if not ANY
  if (caps_.get() != "ANY") {
    GstCaps* caps = gst_caps_from_string(caps_.get().c_str());
    if (caps) {
      gst_app_sink_set_caps(appsink, caps);
      gst_caps_unref(caps);
      HOLOSCAN_LOG_INFO("Set appsink caps: {}", caps_.get());
    } else {
      HOLOSCAN_LOG_ERROR("Failed to parse configured caps: '{}'", caps_.get());
      sink_element_promise_.set_exception(
        std::make_exception_ptr(std::runtime_error("Failed to parse caps"))
      );
      return;
    }
  }

  // Set up appsink callbacks - using static functions for proper ABI compatibility
  // IMPORTANT: Must zero-initialize the entire structure to clear reserved padding fields
  GstAppSinkCallbacks callbacks = {};  // Zero-initialize
  callbacks.eos = appsink_eos_callback;
  callbacks.new_preroll = appsink_new_preroll_callback;
  callbacks.new_sample = appsink_new_sample_callback;

  // Attach callbacks with 'this' as user_data
  gst_app_sink_set_callbacks(appsink, &callbacks, this, NULL);

  // Configure QoS on the base sink
  gst_base_sink_set_qos_enabled(GST_BASE_SINK(appsink), 
                                 static_cast<gboolean>(qos_enabled_.get()));
  
  HOLOSCAN_LOG_INFO("GstSinkResource initialized with appsink (QoS: {}, max_buffers: {})",
                    qos_enabled_.get() ? "enabled" : "disabled", max_buffers_.get());
  
  // Set the promise with the successfully created element
  sink_element_promise_.set_value(std::move(element));
}

std::shared_future<holoscan::gst::Element> GstSinkResource::get_gst_element() const {
  return sink_element_future_;
}

bool GstSinkResource::valid() const {
  return sink_element_future_.valid() && 
         sink_element_future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready && 
         sink_element_future_.get();
}

// Helper functions are now in gst_common.cpp

}  // namespace holoscan
