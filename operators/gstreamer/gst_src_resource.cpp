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
#include "gst/guards.hpp"
#include "gst/buffer.hpp"
#include "gst/caps.hpp"
#include "gst/video_info.hpp"
#include <gst/gst.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <gst/cuda/gstcudamemory.h>
#include <gst/cuda/gstcudacontext.h>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <holoscan/core/execution_context.hpp>
#include <cstring>
#include <memory>

// Convenience constant for mapping CUDA memory for reading
#ifndef GST_MAP_READ_CUDA
#define GST_MAP_READ_CUDA ((GstMapFlags) (GST_MAP_READ | GST_MAP_CUDA))
#endif

// ============================================================================
// Holoscan GstSrcResource Implementation (C++)
// Using standard GStreamer appsrc element instead of custom element
// ============================================================================

namespace holoscan {

  
 /**
  * Abstract base class for wrapping tensor memory into GStreamer memory objects
  */
  class GstSrcResource::MemoryWrapper {
   public:
     virtual ~MemoryWrapper() = default;
     virtual ::GstMemory* wrap_memory(nvidia::gxf::Tensor* tensor, void* user_data, GDestroyNotify notify) = 0;
   };

  /**
 * Host memory wrapper - wraps CPU-accessible memory using standard GStreamer memory
 */
class GstSrcResource::HostMemoryWrapper : public MemoryWrapper {
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
class GstSrcResource::CudaMemoryWrapper : public MemoryWrapper {
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
    cuda_context_ = gst::make_gst_object_guard(gst_cuda_context_new(cuda_device_id));
    if (!cuda_context_) {
      std::string error_msg = fmt::format(
          "Failed to create CUDA context for device {}. "
          "GStreamer CUDA support may not be properly configured.", 
          cuda_device_id);
      HOLOSCAN_LOG_ERROR(error_msg);
      throw std::runtime_error(error_msg);
    }
    
    // Get or create a CUDA allocator - wrap in RAII guard
    GstAllocator* allocator = gst_allocator_find(GST_CUDA_MEMORY_TYPE_NAME);
    if (!allocator) {
      // If not found, create one using g_object_new
      allocator = GST_ALLOCATOR(g_object_new(GST_TYPE_CUDA_ALLOCATOR, nullptr));
    }
    cuda_allocator_ = gst::make_gst_object_guard(allocator);
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
  gst::GstCudaContextGuard cuda_context_;
  gst::GstAllocatorGuard cuda_allocator_;
};

// Push buffer into the pipeline using appsrc
bool GstSrcResource::push_buffer(gst::Buffer buffer, std::chrono::milliseconds timeout) {
  if (!buffer.get()) {
    HOLOSCAN_LOG_ERROR("Invalid buffer provided to push_buffer");
    return false;
  }

  // Check if element is ready
  if (!valid()) {
    HOLOSCAN_LOG_ERROR("Source element not initialized");
    return false;
  }

  GstAppSrc* appsrc = GST_APP_SRC(src_element_future_.get().get());
  
  // Push the buffer to appsrc (transfers ownership)
  // Note: gst_app_src_push_buffer takes ownership and will unref the buffer
  GstFlowReturn ret = gst_app_src_push_buffer(appsrc, gst_buffer_ref(buffer.get()));
  
  if (ret != GST_FLOW_OK) {
    if (ret == GST_FLOW_FLUSHING) {
      HOLOSCAN_LOG_DEBUG("appsrc is flushing, buffer not pushed");
    } else if (ret == GST_FLOW_EOS) {
      HOLOSCAN_LOG_DEBUG("appsrc is in EOS state");
    } else {
      HOLOSCAN_LOG_WARN("Failed to push buffer to appsrc: {}", gst_flow_get_name(ret));
    }
    return false;
  }
  
  HOLOSCAN_LOG_DEBUG("Successfully pushed buffer to appsrc");
  return true;
}

// Get current negotiated caps
gst::Caps GstSrcResource::get_caps() const {
  // Check if element is ready and valid
  if (!valid()) {
    return gst::Caps(); // Return empty caps if not ready
  }

  // Get the source pad and its current caps
  ::GstPad* pad = gst_element_get_static_pad(src_element_future_.get().get(), "src");
  if (!pad) {
    return gst::Caps(); // Return empty caps
  }

  ::GstCaps* caps = gst_pad_get_current_caps(pad);
  gst_object_unref(pad);

  return gst::Caps(caps); // Automatic reference counting
}

// Initialize memory wrapper based on tensor storage type and caps
void GstSrcResource::initialize_memory_wrapper(nvidia::gxf::Tensor* tensor) const {
  // Check if CUDA memory is requested in caps
  std::string caps_str = caps_.get();
  bool cuda_requested = caps_str.find("(memory:CUDAMemory)") != std::string::npos;
  
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

gst::Buffer GstSrcResource::create_buffer_from_entity(const gxf::Entity& entity) const {
  // Create an empty GStreamer buffer at the start (constructor will throw if allocation fails)
  gst::Buffer gst_buffer;

  if (!entity) {
    HOLOSCAN_LOG_ERROR("Invalid entity provided");
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

  int tensor_count = 0;
  size_t total_size = 0;

  // Iterate through all components and process tensors
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
    
    size_t tensor_size = tensor->size();
    void* tensor_data = tensor->pointer();
    
    if (!tensor_data || tensor_size == 0) {
      HOLOSCAN_LOG_WARN("Skipping tensor {} - invalid data or size", tensor_count);
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
      HOLOSCAN_LOG_ERROR("Failed to get DLManagedTensorContext for tensor {}", tensor_count);
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
      HOLOSCAN_LOG_ERROR("Failed to wrap memory for tensor {}", tensor_count);
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
    HOLOSCAN_LOG_ERROR("No valid tensors found in entity");
  } else {
    HOLOSCAN_LOG_DEBUG("Successfully created zero-copy GStreamer buffer from entity: {} tensors, {} total bytes",
                       tensor_count, total_size);
  }
  
  return gst_buffer;
}

GstSrcResource::~GstSrcResource() {
  HOLOSCAN_LOG_INFO("Destroying GstSrcResource");
  
  // Signal EOS to appsrc if element is valid
  if (valid()) {
    GstAppSrc* appsrc = GST_APP_SRC(src_element_future_.get().get());
    gst_app_src_end_of_stream(appsrc);
  }
  
  HOLOSCAN_LOG_INFO("GstSrcResource destroyed");
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
  
  // Initialize the future from the promise (after any construction/moves are complete)
  src_element_future_ = src_element_promise_.get_future();
  
  HOLOSCAN_LOG_INFO("Initializing GstSrcResource with appsrc for data bridging");
  HOLOSCAN_LOG_INFO("Configured capabilities: '{}'", caps_.get());
  
  // Initialize GStreamer if not already done
  if (!gst_is_initialized()) {
    gst_init(nullptr, nullptr);
  }

  // Create appsrc element - standard GStreamer element, no custom registration needed!
  auto element = gst::make_gst_object_guard(
    gst_element_factory_make("appsrc", name().empty() ? nullptr : name().c_str())
  );

  if (!element) {
    HOLOSCAN_LOG_ERROR("Failed to create appsrc element");
    src_element_promise_.set_exception(
      std::make_exception_ptr(std::runtime_error("Failed to create appsrc element"))
    );
    return;
  }

  GstAppSrc* appsrc = GST_APP_SRC(element.get());

  // Configure appsrc properties
  g_object_set(appsrc,
    "stream-type", GST_APP_STREAM_TYPE_STREAM,  // Continuous stream
    "format", GST_FORMAT_TIME,                    // Time-based format
    "is-live", TRUE,                              // Live source (don't wait for timestamps)
    "max-buffers", queue_limit_.get(),            // Buffer queue limit (0 = unlimited)
    "block", FALSE,                               // Don't block push_buffer() when queue is full
    NULL
  );

  // Set caps if not ANY
  if (caps_.get() != "ANY") {
    GstCaps* caps = gst_caps_from_string(caps_.get().c_str());
    if (caps) {
      gst_app_src_set_caps(appsrc, caps);
      gst_caps_unref(caps);
      HOLOSCAN_LOG_INFO("Set appsrc caps: {}", caps_.get());
    } else {
      HOLOSCAN_LOG_ERROR("Failed to parse configured caps: '{}'", caps_.get());
      src_element_promise_.set_exception(
        std::make_exception_ptr(std::runtime_error("Failed to parse caps"))
      );
      return;
    }
  }
  
  HOLOSCAN_LOG_INFO("GstSrcResource initialized with appsrc (queue_limit: {})",
                    queue_limit_.get());
  
  // Set the promise with the successfully created element
  src_element_promise_.set_value(std::move(element));
}

}  // namespace holoscan

