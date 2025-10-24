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

#include "gst_src_bridge.hpp"
#include "gst/video_info.hpp"

#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <gst/cuda/gstcudamemory.h>
#include <gst/cuda/gstcudacontext.h>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <chrono>
#include <thread>
#include <cstring>
#include <memory>
#include <stdexcept>

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

// ============================================================================
// Memory Wrapper Classes
// ============================================================================

/**
 * Abstract base class for wrapping tensor memory into GStreamer memory objects
 */
class GstSrcBridge::MemoryWrapper {
 public:
  virtual ~MemoryWrapper() = default;
  virtual ::GstMemory* wrap_memory(nvidia::gxf::Tensor* tensor, void* user_data, GDestroyNotify notify) = 0;
};

/**
 * Host memory wrapper - wraps CPU-accessible memory using standard GStreamer memory
 */
class GstSrcBridge::HostMemoryWrapper : public MemoryWrapper {
 public:
  HostMemoryWrapper() = default;
  ~HostMemoryWrapper() override = default;
  
  // Non-copyable and non-movable
  HostMemoryWrapper(const HostMemoryWrapper&) = delete;
  HostMemoryWrapper& operator=(const HostMemoryWrapper&) = delete;
  HostMemoryWrapper(HostMemoryWrapper&&) = delete;
  HostMemoryWrapper& operator=(HostMemoryWrapper&&) = delete;
  
  ::GstMemory* wrap_memory(
      nvidia::gxf::Tensor* tensor,
      void* user_data,
      GDestroyNotify notify) override {
    
    void* tensor_data = tensor->pointer();
    size_t tensor_size = tensor->size();
    
    if (!tensor_data || tensor_size == 0) {
      HOLOSCAN_LOG_ERROR("Invalid tensor data or size for host memory wrapping");
      return nullptr;
    }
    
    HOLOSCAN_LOG_DEBUG("Wrapping as host memory (zero-copy): size={} bytes", tensor_size);
    
    return gst_memory_new_wrapped(
        static_cast<GstMemoryFlags>(0),  // flags
        tensor_data,                      // data pointer
        tensor_size,                      // maxsize
        0,                                // offset
        tensor_size,                      // size
        user_data,                        // user_data
        notify);                          // notify callback
  }
};

/**
 * CUDA device memory wrapper - wraps GPU memory using GStreamer CUDA memory
 * Initializes CUDA resources in constructor, throws on failure
 */
class GstSrcBridge::CudaMemoryWrapper : public MemoryWrapper {
 public:
  CudaMemoryWrapper() {
    HOLOSCAN_LOG_INFO("Initializing CUDA resources for zero-copy device memory");
    
    // First, check if CUDA is available using CUDA runtime API
    int device_count = 0;
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count);
    if (cuda_err != cudaSuccess || device_count == 0) {
      std::string error_msg = fmt::format(
          "CUDA not available: {} (device count: {}). "
          "Cannot initialize CUDA resources for zero-copy device memory.",
          cudaGetErrorString(cuda_err), device_count);
      HOLOSCAN_LOG_ERROR(error_msg);
      throw std::runtime_error(error_msg);
    }
    
    HOLOSCAN_LOG_INFO("CUDA detected: {} device(s) available", device_count);
    
    // For now, assume device 0 (can be enhanced to detect actual device from tensor)
    gint cuda_device_id = 0;
    
    // Initialize CUDA memory system (must be called before any GStreamer CUDA operations)
    gst_cuda_memory_init_once();
    
    // Create a CUDA context for this device - wrap in RAII guard
    cuda_context_ = gst::CudaContext(gst_cuda_context_new(cuda_device_id));
    if (!cuda_context_) {
      std::string error_msg = fmt::format(
          "Failed to create CUDA context for device {}. "
          "GStreamer CUDA support may not be properly configured.", 
          cuda_device_id);
      HOLOSCAN_LOG_ERROR(error_msg);
      throw std::runtime_error(error_msg);
    }
    
    // Get or create a CUDA allocator - wrap in RAII guard
    ::GstAllocator* allocator = gst_allocator_find(GST_CUDA_MEMORY_TYPE_NAME);
    if (!allocator) {
      // If not found, create one using g_object_new
      allocator = GST_ALLOCATOR(g_object_new(GST_TYPE_CUDA_ALLOCATOR, nullptr));
    }
    cuda_allocator_ = gst::Allocator(allocator);
    if (!cuda_allocator_) {
      std::string error_msg = "Failed to create CUDA allocator";
      HOLOSCAN_LOG_ERROR(error_msg);
      throw std::runtime_error(error_msg);
    }
    
    HOLOSCAN_LOG_INFO("CUDA resources initialized successfully (device {})", cuda_device_id);
  }
  
  // Non-copyable and non-movable
  CudaMemoryWrapper(const CudaMemoryWrapper&) = delete;
  CudaMemoryWrapper& operator=(const CudaMemoryWrapper&) = delete;
  CudaMemoryWrapper(CudaMemoryWrapper&&) = delete;
  CudaMemoryWrapper& operator=(CudaMemoryWrapper&&) = delete;
  
  ::GstMemory* wrap_memory(
      nvidia::gxf::Tensor* tensor,
      void* user_data,
      GDestroyNotify notify) override {
    
    void* tensor_data = tensor->pointer();
    size_t tensor_size = tensor->size();
    
    if (!tensor_data || tensor_size == 0) {
      HOLOSCAN_LOG_ERROR("Invalid tensor data or size for CUDA memory wrapping");
      return nullptr;
    }
    
    // Get tensor shape to create GstVideoInfo
    auto shape = tensor->shape();
    if (shape.rank() < 2) {
      HOLOSCAN_LOG_ERROR("Tensor has invalid rank {} for CUDA wrapping", shape.rank());
      return nullptr;
    }
    
    // Assume tensor is in format: [height, width, channels] or [height, width]
    gint height = shape.dimension(0);
    gint width = shape.dimension(1);
    GstVideoFormat format = GST_VIDEO_FORMAT_RGBA;  // Assume RGBA for now
    
    // Create video info for the tensor
    GstVideoInfo video_info;
    gst_video_info_init(&video_info);
    if (!gst_video_info_set_format(&video_info, format, width, height)) {
      HOLOSCAN_LOG_ERROR("Failed to set video info for CUDA memory wrapping");
      return nullptr;
    }
    
    HOLOSCAN_LOG_DEBUG("Wrapping as CUDA device memory (zero-copy): size={} bytes", tensor_size);
    
    // Wrap the device memory pointer in GstCudaMemory
    return gst_cuda_allocator_alloc_wrapped(
        GST_CUDA_ALLOCATOR(cuda_allocator_.get()),
        cuda_context_.get(),
        nullptr,                         // CUDA stream (nullptr = default)
        &video_info,                     // video info
        reinterpret_cast<CUdeviceptr>(tensor_data),  // device pointer
        user_data,                       // user_data
        notify);                         // notify callback
  }

 private:
  gst::CudaContext cuda_context_;
  gst::Allocator cuda_allocator_;
};

// ============================================================================
// Tensor Wrapper for Memory Lifetime Management
// ============================================================================

// Wrapper to keep tensor alive while GStreamer uses its memory
struct TensorWrapper {
  std::shared_ptr<nvidia::gxf::DLManagedTensorContext> dl_ctx;  // Keep tensor memory alive
  
  explicit TensorWrapper(std::shared_ptr<nvidia::gxf::DLManagedTensorContext> ctx) 
    : dl_ctx(std::move(ctx)) {}
};

// Callback to free TensorWrapper when GstMemory is destroyed
static void free_tensor_wrapper(gpointer user_data) {
  auto* wrapper = static_cast<TensorWrapper*>(user_data);
  delete wrapper;
}

// ============================================================================
// GstSrcBridge Implementation
// ============================================================================

GstSrcBridge::GstSrcBridge(const std::string& name, const std::string& caps, size_t max_buffers)
    : name_(name), 
      caps_(caps), 
      max_buffers_(max_buffers),
      src_element_{gst_element_factory_make("appsrc", name_.empty() ? nullptr : name_.c_str())} {
  HOLOSCAN_LOG_INFO("Creating GstSrcBridge: name='{}', caps='{}', max_buffers={}",
                    name, caps, max_buffers);
  
  if (!src_element_) {
    HOLOSCAN_LOG_ERROR("Failed to create appsrc element");
    throw std::runtime_error("Failed to create appsrc element");
  }

  GstAppSrc* appsrc = GST_APP_SRC(src_element_.get());

  // Parse framerate from caps first to determine is-live mode
  bool is_live = false;
  if (caps_ != "ANY") {
    GstCaps* gst_caps = gst_caps_from_string(caps_.c_str());
    if (gst_caps) {
      // Extract framerate from caps
      if (gst_caps_get_size(gst_caps) > 0) {
        GstStructure* structure = gst_caps_get_structure(gst_caps, 0);
        const GValue* framerate_value = gst_structure_get_value(structure, "framerate");
        
        if (framerate_value && GST_VALUE_HOLDS_FRACTION(framerate_value)) {
          framerate_num_ = gst_value_get_fraction_numerator(framerate_value);
          framerate_den_ = gst_value_get_fraction_denominator(framerate_value);
          
          // If framerate is 0, treat as live source (process frames as fast as they come)
          if (framerate_num_ == 0) {
            is_live = true;
            HOLOSCAN_LOG_INFO("Framerate is 0/1 - using live mode (no framerate control)");
          } else {
            HOLOSCAN_LOG_INFO("Parsed framerate from caps: {}/{} fps", 
                              framerate_num_, framerate_den_);
          }
        } else {
          // No framerate specified - use live mode
          is_live = true;
          framerate_num_ = 0;
          framerate_den_ = 1;
          HOLOSCAN_LOG_INFO("Framerate not found in caps - using live mode (no framerate control)");
        }
      }
      gst_caps_unref(gst_caps);
    }
  }

  // Configure appsrc properties
  g_object_set(appsrc,
    "stream-type", GST_APP_STREAM_TYPE_STREAM,  // Continuous stream
    "format", GST_FORMAT_TIME,                    // Time-based format
    "is-live", is_live ? TRUE : FALSE,            // Live mode if framerate is 0 or not specified
    "max-buffers", max_buffers_,                  // Buffer queue limit (0 = unlimited)
    "max-bytes", (guint64)0,                      // Byte limit (0 = unlimited, controlled by max-buffers)
    "block", TRUE,                                // Block push_buffer() when queue is full for proper flow control
    NULL
  );
  
  HOLOSCAN_LOG_INFO("appsrc configured: max-buffers={}, is-live={}, block=true", 
                    max_buffers_, is_live ? "true" : "false");

  // Set caps if not ANY
  if (caps_ != "ANY") {
    GstCaps* gst_caps = gst_caps_from_string(caps_.c_str());
    if (gst_caps) {
      gst_app_src_set_caps(appsrc, gst_caps);
      gst_caps_unref(gst_caps);
      HOLOSCAN_LOG_INFO("Set appsrc caps: {}", caps_);
    } else {
      HOLOSCAN_LOG_ERROR("Failed to parse configured caps: '{}'", caps_);
      throw std::runtime_error("Failed to parse caps");
    }
  }
  
  HOLOSCAN_LOG_INFO("GstSrcBridge initialized with appsrc (max_buffers: {}, framerate: {}/{})", 
                    max_buffers_, framerate_num_, framerate_den_);
}

GstSrcBridge::~GstSrcBridge() {
  HOLOSCAN_LOG_INFO("Destroying GstSrcBridge");
  
  // Attempt to send EOS during destruction
  if (send_eos()) {
    HOLOSCAN_LOG_INFO("Sent EOS during destruction");
  }
  
  HOLOSCAN_LOG_INFO("GstSrcBridge destroyed");
}

bool GstSrcBridge::push_buffer(gst::Buffer buffer, std::chrono::milliseconds timeout) {
  HOLOSCAN_LOG_DEBUG("GstSrcBridge::push_buffer() - Starting");
  
  if (!buffer.get()) {
    HOLOSCAN_LOG_ERROR("Invalid buffer provided to push_buffer");
    return false;
  }

  GstAppSrc* appsrc = GST_APP_SRC(src_element_.get());
  
  // Check appsrc queue status
  guint64 current_level_bytes = 0;
  guint64 max_bytes = 0;
  guint current_level_buffers = 0;
  guint max_buffers = 0;
  g_object_get(appsrc, 
               "current-level-bytes", &current_level_bytes, 
               "max-bytes", &max_bytes,
               "current-level-buffers", &current_level_buffers,
               "max-buffers", &max_buffers,
               NULL);
  HOLOSCAN_LOG_DEBUG("appsrc queue: {}/{} buffers, {}/{} bytes", 
                    current_level_buffers, max_buffers,
                    current_level_bytes, max_bytes);
  
  HOLOSCAN_LOG_DEBUG("Calling gst_app_src_push_buffer()");
  
  // Push the buffer to appsrc (transfers ownership)
  // Note: gst_app_src_push_buffer takes ownership and will unref the buffer
  GstFlowReturn ret = gst_app_src_push_buffer(appsrc, gst_buffer_ref(buffer.get()));
  
  HOLOSCAN_LOG_DEBUG("gst_app_src_push_buffer() returned: {} ({})", 
                    gst_flow_get_name(ret), static_cast<int>(ret));
  
  if (ret != GST_FLOW_OK) {
    if (ret == GST_FLOW_FLUSHING) {
      HOLOSCAN_LOG_WARN("appsrc is flushing, buffer not pushed");
    } else if (ret == GST_FLOW_EOS) {
      HOLOSCAN_LOG_WARN("appsrc is in EOS state");
    } else {
      HOLOSCAN_LOG_ERROR("Failed to push buffer to appsrc: {}", gst_flow_get_name(ret));
    }
    return false;
  }
  
  HOLOSCAN_LOG_DEBUG("Successfully pushed buffer to appsrc");
  return true;
}

bool GstSrcBridge::send_eos() {
  GstAppSrc* appsrc = GST_APP_SRC(src_element_.get());
  
  // Send EOS to appsrc
  // gst_app_src_end_of_stream() handles draining internally - it ensures all queued
  // buffers are processed before sending EOS downstream
  // The caller should wait for the EOS message on the pipeline bus for proper synchronization
  HOLOSCAN_LOG_INFO("Sending EOS to appsrc to finalize stream");
  GstFlowReturn ret = gst_app_src_end_of_stream(appsrc);
  
  switch (ret) {
    case GST_FLOW_OK:
      HOLOSCAN_LOG_INFO("Successfully sent EOS to appsrc");
      break;
    case GST_FLOW_EOS:
      HOLOSCAN_LOG_DEBUG("EOS already sent, ignoring duplicate call");
      break;
    default:
      HOLOSCAN_LOG_WARN("Failed to send EOS: {}", gst_flow_get_name(ret));
      break;
  }
  
  return ret == GST_FLOW_OK;
}

gst::Caps GstSrcBridge::get_caps() const {
  // Get the source pad and its current caps
  ::GstPad* pad = gst_element_get_static_pad(src_element_.get(), "src");
  if (!pad) {
    return gst::Caps(); // Return empty caps
  }

  ::GstCaps* caps = gst_pad_get_current_caps(pad);
  gst_object_unref(pad);

  return gst::Caps(caps); // Automatic reference counting
}

void GstSrcBridge::initialize_memory_wrapper(nvidia::gxf::Tensor* tensor) {
  // Check if CUDA memory is requested in caps
  bool cuda_requested = caps_.find("(memory:CUDAMemory)") != std::string::npos;
  
  const char* storage_type_str = (tensor->storage_type() == nvidia::gxf::MemoryStorageType::kDevice) 
      ? "GPU" : "CPU";
  
  // Use CudaMemoryWrapper for GPU tensors when GStreamer requests CUDA memory
  if (tensor->storage_type() == nvidia::gxf::MemoryStorageType::kDevice && cuda_requested) {
    HOLOSCAN_LOG_INFO("Creating CUDA memory wrapper ({} memory)", storage_type_str);
    memory_wrapper_.reset(new CudaMemoryWrapper());
  } else {
    // Use HostMemoryWrapper for CPU memory or when CUDA not requested
    HOLOSCAN_LOG_INFO("Creating host memory wrapper ({} memory)", storage_type_str);
    memory_wrapper_.reset(new HostMemoryWrapper());
  }
}

gst::Buffer GstSrcBridge::create_buffer_from_tensors(nvidia::gxf::Tensor** tensors, size_t num_tensors) {
  // Create an empty GStreamer buffer at the start (constructor will throw if allocation fails)
  gst::Buffer gst_buffer;

  if (!tensors || num_tensors == 0) {
    HOLOSCAN_LOG_ERROR("Invalid tensors array or count");
    return gst_buffer;
  }

  int tensor_count = 0;
  size_t total_size = 0;

  // Iterate through all tensors
  for (size_t i = 0; i < num_tensors; i++) {
    auto* tensor = tensors[i];
    if (!tensor) {
      HOLOSCAN_LOG_WARN("Skipping null tensor at index {}", i);
      continue;
    }
    
    size_t tensor_size = tensor->size();
    void* tensor_data = tensor->pointer();
    
    if (!tensor_data || tensor_size == 0) {
      HOLOSCAN_LOG_WARN("Skipping tensor {} - invalid data or size", i);
      continue;
    }

    // Lazy initialization of memory wrapper on first tensor
    if (!memory_wrapper_) {
      try {
        initialize_memory_wrapper(tensor);
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Failed to create memory wrapper: {}", e.what());
        return gst_buffer;
      }
    }

    // Get the DLManagedTensorContext shared_ptr to keep tensor memory alive
    auto maybe_dl_ctx = tensor->toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
      HOLOSCAN_LOG_ERROR("Failed to get DLManagedTensorContext for tensor {}", i);
      continue;
    }

    // Create a TensorWrapper with the shared_ptr to keep tensor alive
    auto tensor_wrapper = std::make_unique<TensorWrapper>(maybe_dl_ctx.value());

    // Use the memory wrapper to wrap the tensor
    ::GstMemory* memory = memory_wrapper_->wrap_memory(
        tensor,
        tensor_wrapper.get(),
        free_tensor_wrapper);
    
    if (!memory) {
      HOLOSCAN_LOG_ERROR("Failed to wrap memory for tensor {}", i);
      continue;
    }

    // Release ownership - GStreamer now manages the wrapper lifetime
    tensor_wrapper.release();

    // Append wrapped memory to buffer
    gst_buffer_append_memory(gst_buffer.get(), memory);
    
    tensor_count++;
    total_size += tensor_size;
  }

  if (tensor_count == 0) {
    HOLOSCAN_LOG_ERROR("No valid tensors found");
  } else {
    // Set timestamps on the buffer based on frame count and framerate
    GstClockTime timestamp;
    GstClockTime duration;
    
    if (framerate_num_ == 0) {
      // Live mode: use real-time timestamps (current monotonic time)
      // This reflects actual frame arrival times rather than synthetic framerate
      timestamp = g_get_monotonic_time() * 1000;  // Convert microseconds to nanoseconds
      duration = GST_CLOCK_TIME_NONE;  // Duration unknown in live mode
    } else {
      // Calculate timestamp directly from frame count to avoid accumulating rounding errors
      // timestamp = (frame_count * denom * GST_SECOND) / num
      timestamp = gst_util_uint64_scale(frame_count_, 
                                        framerate_den_ * GST_SECOND, 
                                        framerate_num_);
      
      // Calculate duration as difference between next and current timestamp
      // duration = timestamp(frame_count + 1) - timestamp(frame_count)
      GstClockTime next_timestamp = gst_util_uint64_scale(frame_count_ + 1, 
                                                           framerate_den_ * GST_SECOND, 
                                                           framerate_num_);
      duration = next_timestamp - timestamp;
    }
    
    GST_BUFFER_PTS(gst_buffer.get()) = timestamp;
    GST_BUFFER_DTS(gst_buffer.get()) = timestamp;
    GST_BUFFER_DURATION(gst_buffer.get()) = duration;
    
    frame_count_++;
    
    HOLOSCAN_LOG_DEBUG("Successfully created zero-copy GStreamer buffer: {} tensors, {} total bytes, frame={}, PTS={}",
                       tensor_count, total_size, frame_count_ - 1, 
                       (timestamp == GST_CLOCK_TIME_NONE) ? "NONE" : std::to_string(timestamp) + " ns");
  }
  
  return gst_buffer;
}

gst::Buffer GstSrcBridge::create_buffer_from_entity(const nvidia::gxf::Entity& entity) {
  // Create an empty GStreamer buffer at the start
  gst::Buffer gst_buffer;

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

  // Delegate to create_buffer_from_tensors
  return create_buffer_from_tensors(tensors.data(), tensors.size());
}

}  // namespace holoscan

