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

#include "gst_sink_bridge.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/base/gstbasesink.h>
#include <gst/video/video.h>
#include <gst/cuda/gstcudamemory.h>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <gxf/std/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/logger/logger.hpp>

// Convenience constant for mapping CUDA memory for reading
#ifndef GST_MAP_READ_CUDA
#define GST_MAP_READ_CUDA ((GstMapFlags) (GST_MAP_READ | GST_MAP_CUDA))
#endif

namespace holoscan {

static bool gst_initialized = [](){
  // Initialize GStreamer if not already done
  if (!gst_is_initialized()) {
    gst_init(nullptr, nullptr);
  }
  return true;
}();

// Forward declare callback functions (defined below)
void appsink_eos_callback(::GstAppSink* appsink, gpointer user_data);
::GstFlowReturn appsink_new_preroll_callback(::GstAppSink* appsink, gpointer user_data);
::GstFlowReturn appsink_new_sample_callback(::GstAppSink* appsink, gpointer user_data);

namespace {

/**
 * @brief Tensor metadata (name, shape, strides) - internal implementation detail
 */
struct TensorMetadata {
  std::string name;
  nvidia::gxf::Shape shape;
  std::array<size_t, 8> strides;
};

/**
 * @brief Get tensor metadata for video plane
 */
TensorMetadata get_video_tensor_metadata(
    const gst::VideoInfo& video_info,
    guint plane_idx,
    guint n_planes) {
  
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

/**
 * @brief Get tensor metadata for generic memory block
 */
TensorMetadata get_generic_tensor_metadata(
    guint mem_idx,
    gsize size,
    guint n_mem) {
  
  // Create tensor name
  std::string tensor_name = (n_mem == 1) ? "data" : fmt::format("data_{}", mem_idx);
  
  // Create shape: 1D byte array
  nvidia::gxf::Shape shape({static_cast<int32_t>(size)});
  
  // Strides: just element size
  std::array<size_t, 8> strides{{1, 0, 0, 0, 0, 0, 0, 0}};
  
  return {tensor_name, shape, strides};
}

/**
 * @brief Helper function to map GStreamer memory with the specified flags
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
// Callback functions for appsink
// ============================================================================

void appsink_eos_callback(::GstAppSink* appsink, gpointer user_data) {
  GstSinkBridge* bridge = static_cast<GstSinkBridge*>(user_data);
  HOLOSCAN_LOG_INFO("End of stream in appsink: {}", bridge->name_);
}

::GstFlowReturn appsink_new_preroll_callback(::GstAppSink* appsink, gpointer user_data) {
  GstSinkBridge* bridge = static_cast<GstSinkBridge*>(user_data);
  
  // Check if shutting down before pulling sample
  {
    std::unique_lock<std::mutex> lock(bridge->mutex_);
    if (bridge->is_shutting_down_) {
      return GST_FLOW_EOS;
    }
  }
  
  ::GstSample* sample = gst_app_sink_pull_preroll(appsink);
  if (!sample) {
    return GST_FLOW_ERROR;
  }
  
  ::GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    gst_sample_unref(sample);
    return GST_FLOW_ERROR;
  }
  
  HOLOSCAN_LOG_DEBUG("Received preroll buffer via appsink, size: {} bytes", 
                     gst_buffer_get_size(buffer));
  
  std::unique_lock<std::mutex> lock(bridge->mutex_);
  
  // Double-check shutdown flag after acquiring lock
  if (bridge->is_shutting_down_) {
    gst_sample_unref(sample);
    return GST_FLOW_EOS;
  }
  
  gst::Buffer buffer_obj(gst_buffer_ref(buffer));
  
  if (bridge->pending_request_.has_value()) {
    std::promise<gst::Buffer> promise = std::move(bridge->pending_request_.value());
    bridge->pending_request_.reset();
    promise.set_value(std::move(buffer_obj));
    HOLOSCAN_LOG_DEBUG("Fulfilled pending buffer request with preroll");
  } else {
    bridge->buffer_queue_.push(std::move(buffer_obj));
    HOLOSCAN_LOG_DEBUG("Queued preroll buffer, total in queue: {}", 
                       bridge->buffer_queue_.size());
  }
  
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

::GstFlowReturn appsink_new_sample_callback(::GstAppSink* appsink, gpointer user_data) {
  GstSinkBridge* bridge = static_cast<GstSinkBridge*>(user_data);
  
  // Check if shutting down before pulling sample
  {
    std::unique_lock<std::mutex> lock(bridge->mutex_);
    if (bridge->is_shutting_down_) {
      return GST_FLOW_EOS;
    }
  }
  
  // Pull the sample (contains buffer + caps)
  ::GstSample* sample = gst_app_sink_pull_sample(appsink);
  if (!sample) {
    return GST_FLOW_ERROR;
  }
  
  // Extract buffer from sample
  ::GstBuffer* buffer = gst_sample_get_buffer(sample);
  if (!buffer) {
    gst_sample_unref(sample);
    return GST_FLOW_ERROR;
  }
  
  HOLOSCAN_LOG_DEBUG("Received buffer via appsink, size: {} bytes", 
                     gst_buffer_get_size(buffer));
  
  // Lock for thread-safe queue access
  std::unique_lock<std::mutex> lock(bridge->mutex_);
  
  // Double-check shutdown flag after acquiring lock
  if (bridge->is_shutting_down_) {
    gst_sample_unref(sample);
    return GST_FLOW_EOS;
  }
  
  // Create Buffer object (ref the buffer before unreffing sample)
  gst::Buffer buffer_obj(gst_buffer_ref(buffer));
  
  // Check if there is a pending request waiting for a buffer
  if (bridge->pending_request_.has_value()) {
    // Fulfill the pending request immediately
    std::promise<gst::Buffer> promise = std::move(bridge->pending_request_.value());
    bridge->pending_request_.reset();
    promise.set_value(std::move(buffer_obj));
    HOLOSCAN_LOG_DEBUG("Fulfilled pending buffer request");
  } else {
    // Queue the buffer for future requests
    bridge->buffer_queue_.push(std::move(buffer_obj));
    HOLOSCAN_LOG_DEBUG("Queued buffer, total in queue: {}", bridge->buffer_queue_.size());
    
    // Backpressure: wait if queue is full
    size_t max_buffers = bridge->max_buffers_;
    if (bridge->buffer_queue_.size() > max_buffers) {
      HOLOSCAN_LOG_DEBUG("Buffer queue size ({}) exceeds limit ({}), waiting...", 
                        bridge->buffer_queue_.size(), max_buffers);
      
      bridge->queue_cv_.wait(lock, [bridge, max_buffers]() {
        return bridge->is_shutting_down_ || bridge->buffer_queue_.size() <= max_buffers;
      });
      
      // Check if we woke up due to shutdown
      if (bridge->is_shutting_down_) {
        gst_sample_unref(sample);
        return GST_FLOW_EOS;
      }
      
      HOLOSCAN_LOG_DEBUG("Buffer queue size now within limit: {}", 
                        bridge->buffer_queue_.size());
    }
  }
  
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

// ============================================================================
// GstSinkBridge Implementation
// ============================================================================

GstSinkBridge::GstSinkBridge(const std::string& name, const std::string& caps,
                             size_t max_buffers, bool qos_enabled)
    : name_(name),
      caps_(caps),
      qos_enabled_(qos_enabled),
      max_buffers_(max_buffers),
      sink_element_(gst_element_factory_make("appsink", name_.empty() ? nullptr : name_.c_str())),
      is_shutting_down_(false) {
  
  HOLOSCAN_LOG_INFO("Creating GstSinkBridge: name='{}', caps='{}', max_buffers={}, qos={}",
                    name, caps, max_buffers, qos_enabled ? "enabled" : "disabled");
  
  if (!sink_element_) {
    HOLOSCAN_LOG_ERROR("Failed to create appsink element");
    throw std::runtime_error("Failed to create appsink element");
  }

  ::GstAppSink* appsink = GST_APP_SINK(sink_element_.get());

  // Configure appsink properties
  g_object_set(appsink,
    "emit-signals", FALSE,  // Use callbacks instead of signals (more efficient)
    "sync", TRUE,           // Sync to clock for proper timing
    "max-buffers", static_cast<guint>(max_buffers_ + 1),  // Buffer queue limit
    "drop", FALSE,          // Don't drop buffers - we handle backpressure
    NULL
  );

  // Set caps if not ANY
  if (caps_ != "ANY") {
    ::GstCaps* gst_caps = gst_caps_from_string(caps_.c_str());
    if (gst_caps) {
      gst_app_sink_set_caps(appsink, gst_caps);
      gst_caps_unref(gst_caps);
      HOLOSCAN_LOG_INFO("Set appsink caps: {}", caps_);
    } else {
      HOLOSCAN_LOG_ERROR("Failed to parse configured caps: '{}'", caps_);
      throw std::runtime_error("Failed to parse caps");
    }
  }

  // Set up appsink callbacks
  ::GstAppSinkCallbacks callbacks = {};  // Zero-initialize
  callbacks.eos = appsink_eos_callback;
  callbacks.new_preroll = appsink_new_preroll_callback;
  callbacks.new_sample = appsink_new_sample_callback;

  // Attach callbacks with 'this' as user_data
  gst_app_sink_set_callbacks(appsink, &callbacks, this, NULL);

  // Configure QoS on the base sink
  gst_base_sink_set_qos_enabled(GST_BASE_SINK(appsink), 
                                 static_cast<gboolean>(qos_enabled_));
  
  HOLOSCAN_LOG_INFO("GstSinkBridge initialized with appsink (max_buffers: {}, QoS: {})",
                    max_buffers_, qos_enabled_ ? "enabled" : "disabled");
}

GstSinkBridge::~GstSinkBridge() {
  HOLOSCAN_LOG_INFO("Destroying GstSinkBridge");
  
  // Signal shutdown and wake up any waiting callbacks
  {
    std::lock_guard<std::mutex> lock(mutex_);
    is_shutting_down_ = true;
    std::queue<gst::Buffer> empty_queue;
    std::swap(buffer_queue_, empty_queue);
    pending_request_.reset();
    queue_cv_.notify_all();
  }
  
  HOLOSCAN_LOG_INFO("GstSinkBridge destroyed");
}

std::future<gst::Buffer> GstSinkBridge::pop_buffer() {
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
    HOLOSCAN_LOG_DEBUG("Fulfilled buffer request immediately, remaining buffers: {}", 
                       buffer_queue_.size());
    
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

gst::Caps GstSinkBridge::get_caps() const {
  // Get the sink pad and its current caps
  ::GstPad* pad = gst_element_get_static_pad(sink_element_.get(), "sink");
  if (!pad) {
    return gst::Caps(); // Return empty caps
  }

  ::GstCaps* caps = gst_pad_get_current_caps(pad);
  gst_object_unref(pad);

  return gst::Caps(caps); // Automatic reference counting
}

gxf::Entity GstSinkBridge::create_entity_from_buffer(
    ExecutionContext& context,
    const gst::Buffer& buffer) const {
  
  // Get current caps
  gst::Caps caps = get_caps();
  
  // Get the GXF context
  gxf_context_t gxf_context = context.context();
  
  // Create entity to hold tensor(s)
  auto entity_result = nvidia::gxf::Entity::New(gxf_context);
  if (!entity_result) {
    HOLOSCAN_LOG_ERROR("Failed to create entity");
    return gxf::Entity();
  }
  auto entity = entity_result.value();

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
    // Check if caps indicate CUDA memory, if so, try CUDA mapping first
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
    auto gxf_tensor_result = entity.add<nvidia::gxf::Tensor>(metadata.name.c_str());
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
  
  return gxf::Entity(std::move(entity));
}

}  // namespace holoscan

