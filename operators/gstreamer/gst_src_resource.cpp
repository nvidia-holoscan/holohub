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
#include <gst/base/gstpushsrc.h>
#include <gst/video/video.h>
#include <gst/cuda/gstcudamemory.h>
#include <gst/cuda/gstcudacontext.h>

#include <cuda_runtime_api.h>
#include <dlpack/dlpack.h>
#include <holoscan/core/execution_context.hpp>
#include <cstring>
#include <memory>

// Forward declaration of GstSrcResource for C code
namespace holoscan { class GstSrcResource; }

// Convenience constant for mapping CUDA memory for reading
#ifndef GST_MAP_READ_CUDA
#define GST_MAP_READ_CUDA ((GstMapFlags) (GST_MAP_READ | GST_MAP_CUDA))
#endif

extern "C" {

// ============================================================================
// GStreamer Custom Push Source Element Implementation (embedded in C++)
// ============================================================================

/* Standard macros for defining the GStreamer element */
#define GST_TYPE_HOLOSCAN_SRC \
  (gst_holoscan_src_get_type())
#define GST_HOLOSCAN_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_HOLOSCAN_SRC,GstHoloscanSrc))
#define GST_HOLOSCAN_SRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_HOLOSCAN_SRC,GstHoloscanSrcClass))
#define GST_IS_HOLOSCAN_SRC(obj) \
  (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_HOLOSCAN_SRC))
#define GST_IS_HOLOSCAN_SRC_CLASS(klass) \
  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_HOLOSCAN_SRC))

typedef struct _GstHoloscanSrc GstHoloscanSrc;
typedef struct _GstHoloscanSrcClass GstHoloscanSrcClass;

/**
 * GstHoloscanSrc:
 * @parent: the parent object
 * @caps_set: whether caps have been negotiated
 * @caps: the negotiated caps for this source
 * @holoscan_resource: pointer back to the GstSrcResource instance
 *
 * The Holoscan source object structure (internal GStreamer element for data bridging)
 */
struct _GstHoloscanSrc
{
  GstPushSrc parent;

  /* Processing state */
  gboolean caps_set;

  /* Media information */
  GstCaps *caps;          // Full caps information

  /* Bridge to C++ Holoscan resource */
  void* holoscan_resource;  // GstSrcResource* (stored as void* for C compatibility)
};

/**
 * GstHoloscanSrcClass:
 * @parent_class: the parent class
 *
 * The Holoscan source class structure (internal GStreamer element class)
 */
struct _GstHoloscanSrcClass
{
  GstPushSrcClass parent_class;
};

GST_DEBUG_CATEGORY_STATIC(gst_holoscan_src_debug);
#define GST_CAT_DEFAULT gst_holoscan_src_debug

/* Pad templates - will be dynamically updated based on configured caps */
static GstStaticPadTemplate src_pad_template = GST_STATIC_PAD_TEMPLATE("src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("ANY")  // Default fallback, will be overridden
);

/* Function prototypes */
static void gst_holoscan_src_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec);
static void gst_holoscan_src_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec);
static void gst_holoscan_src_finalize(GObject *object);

/* Helper function to extract media type from caps */
static const gchar* gst_holoscan_src_get_media_type_string(GstCaps *caps);

/* Helper function implementations */
static const gchar* gst_holoscan_src_get_media_type_string(GstCaps *caps)
{
  if (!caps || gst_caps_is_empty(caps)) {
    return "unknown";
  }

  if (gst_caps_is_any(caps)) {
    return "ANY";
  }

  GstStructure *structure = gst_caps_get_structure(caps, 0);
  if (!structure) {
    return "unknown";
  }

  return gst_structure_get_name(structure);
}

/* Class initialization */
#define gst_holoscan_src_parent_class parent_class
G_DEFINE_TYPE(GstHoloscanSrc, gst_holoscan_src, GST_TYPE_PUSH_SRC);

static void
gst_holoscan_src_class_init(GstHoloscanSrcClass *klass)
{
  GObjectClass *gobject_class;
  GstElementClass *gstelement_class;
  GstBaseSrcClass *gstbasesrc_class;
  GstPushSrcClass *gstpushsrc_class;

  gobject_class = G_OBJECT_CLASS(klass);
  gstelement_class = GST_ELEMENT_CLASS(klass);
  gstbasesrc_class = GST_BASE_SRC_CLASS(klass);
  gstpushsrc_class = GST_PUSH_SRC_CLASS(klass);

  /* Set up object methods */
  gobject_class->set_property = gst_holoscan_src_set_property;
  gobject_class->get_property = gst_holoscan_src_get_property;
  gobject_class->finalize = gst_holoscan_src_finalize;

  /* No properties needed for basic data bridging */

  /* Set element metadata */
  gst_element_class_set_static_metadata(gstelement_class,
      "Holoscan Bridge Source",
      "Source/Generic",
      "A GStreamer source element that bridges data from Holoscan operators",
      "NVIDIA Corporation <holoscan@nvidia.com>");

  /* Add pad template */
  gst_element_class_add_static_pad_template(gstelement_class, &src_pad_template);

  /* Set up base source methods using static member functions */
  gstbasesrc_class->get_caps = GST_DEBUG_FUNCPTR(holoscan::GstSrcResource::get_caps_callback);
  gstbasesrc_class->set_caps = GST_DEBUG_FUNCPTR(holoscan::GstSrcResource::set_caps_callback);
  gstbasesrc_class->start = GST_DEBUG_FUNCPTR(holoscan::GstSrcResource::start_callback);
  gstbasesrc_class->stop = GST_DEBUG_FUNCPTR(holoscan::GstSrcResource::stop_callback);
  gstpushsrc_class->create = GST_DEBUG_FUNCPTR(holoscan::GstSrcResource::create_callback);

  /* Initialize debug category */
  GST_DEBUG_CATEGORY_INIT(gst_holoscan_src_debug, "holoscansrc", 0,
      "Holoscan Source Element");
}

static void
gst_holoscan_src_init(GstHoloscanSrc *src)
{
  /* Initialize state */
  src->caps_set = FALSE;
  src->holoscan_resource = NULL;

  /* Initialize caps */
  src->caps = NULL;

  /* Configure as live source */
  gst_base_src_set_live(GST_BASE_SRC(src), TRUE);
  gst_base_src_set_format(GST_BASE_SRC(src), GST_FORMAT_TIME);
}

static void
gst_holoscan_src_finalize(GObject *object)
{
  GstHoloscanSrc *src = GST_HOLOSCAN_SRC(object);

  /* Clean up caps information */
  if (src->caps) {
    gst_caps_unref(src->caps);
  }

  HOLOSCAN_LOG_DEBUG("Finalizing Holoscan source");

  G_OBJECT_CLASS(parent_class)->finalize(object);
}

static void
gst_holoscan_src_set_property(GObject *object, guint prop_id,
    const GValue *value, GParamSpec *pspec)
{
  GstHoloscanSrc *src = GST_HOLOSCAN_SRC(object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

static void
gst_holoscan_src_get_property(GObject *object, guint prop_id,
    GValue *value, GParamSpec *pspec)
{
  GstHoloscanSrc *src = GST_HOLOSCAN_SRC(object);

  switch (prop_id) {
    default:
      G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
      break;
  }
}

/* Element registration function for direct use */
gboolean
gst_holoscan_src_plugin_init(GstPlugin *plugin)
{
  return gst_element_register(plugin, "holoscansrc", GST_RANK_NONE,
      GST_TYPE_HOLOSCAN_SRC);
}

}  // extern "C"

// ============================================================================
// Holoscan GstSrcResource Implementation (C++)
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

// Push buffer into the pipeline
bool GstSrcResource::push_buffer(gst::Buffer buffer, std::chrono::milliseconds timeout) {
  
  std::unique_lock<std::mutex> lock(mutex_);
  
  // Wait if queue is at capacity (only if queue_limit is not 0)
  size_t limit = queue_limit_.get();
  
  // Wait for space in queue with timeout (0 means try immediately and return)
  if (!queue_cv_.wait_for(lock, timeout, [this, limit]() {
    auto resource = GST_HOLOSCAN_SRC(src_element_future_.get().get())->holoscan_resource;
    return resource == nullptr || buffer_queue_.size() <= limit;
  })) {
    HOLOSCAN_LOG_WARN("push_buffer failed: queue full (timeout: {} ms)", timeout.count());
    return false;
  }
  
  auto resource = GST_HOLOSCAN_SRC(src_element_future_.get().get())->holoscan_resource;
  // Check if we woke up due to shutdown
  if (resource == nullptr) {
    HOLOSCAN_LOG_INFO("GstSrcResource destroyed");
    return false;
  }
  
  buffer_queue_.emplace(std::move(buffer));
  queue_cv_.notify_one();
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
  
  // Determine which wrapper to create based on tensor storage type and caps
  if (tensor->storage_type() == nvidia::gxf::MemoryStorageType::kDevice && cuda_requested) {
    HOLOSCAN_LOG_INFO("First CUDA tensor detected - creating CUDA memory wrapper");
    memory_wrapper_.reset(new CudaMemoryWrapper());
  } else {
    HOLOSCAN_LOG_INFO("Creating host memory wrapper");
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

// Static member function implementations for GStreamer callbacks

// Get caps callback - advertise what the source will provide
::GstCaps* GstSrcResource::get_caps_callback(::GstBaseSrc *src, ::GstCaps *filter) {
  GstHoloscanSrc *holoscan_src = GST_HOLOSCAN_SRC(src);
  
  /* Access the GstSrcResource instance to get the configured caps */
  if (holoscan_src->holoscan_resource) {
    GstSrcResource* resource = static_cast<GstSrcResource*>(holoscan_src->holoscan_resource);
    std::string configured_caps = resource->caps_.get();
    
    if (configured_caps != "ANY") {
      /* Create GstCaps from the configured caps string */
      GstCaps *caps = gst_caps_from_string(configured_caps.c_str());
      if (caps) {
        HOLOSCAN_LOG_DEBUG("Advertising source capabilities: {}", configured_caps);
        
        /* If there's a filter, intersect with it */
        if (filter) {
          GstCaps *filtered_caps = gst_caps_intersect_full(caps, filter, GST_CAPS_INTERSECT_FIRST);
          gst_caps_unref(caps);
          return filtered_caps;
        }
        
        return caps;
      } else {
        HOLOSCAN_LOG_ERROR("Failed to parse configured caps: '{}'", configured_caps);
      }
    }
  }
  
  /* Fallback to template caps */
  GstPad *pad = GST_BASE_SRC_PAD(src);
  GstCaps *template_caps = gst_pad_get_pad_template_caps(pad);
  
  if (filter) {
    GstCaps *filtered_caps = gst_caps_intersect_full(template_caps, filter, GST_CAPS_INTERSECT_FIRST);
    gst_caps_unref(template_caps);
    return filtered_caps;
  }
  
  return template_caps;
}

// Set caps callback
gboolean GstSrcResource::set_caps_callback(::GstBaseSrc *src, ::GstCaps *caps) {
  GstHoloscanSrc *holoscan_src = GST_HOLOSCAN_SRC(src);
  const gchar *media_type;

  /* Get media type using our helper function */
  media_type = gst_holoscan_src_get_media_type_string(caps);

  /* Access the GstSrcResource instance to get the configured caps */
  if (holoscan_src->holoscan_resource) {
    GstSrcResource* resource = static_cast<GstSrcResource*>(holoscan_src->holoscan_resource);
    std::string configured_caps = resource->caps_.get();
    
    HOLOSCAN_LOG_INFO("Setting caps: {} (configured: '{}', incoming: {})", 
        gst_caps_to_string(caps), configured_caps, media_type);
    
    /* Accept the caps */
    if (configured_caps != "ANY") {
      HOLOSCAN_LOG_INFO("Accepting caps: {} (source configured for: '{}')", 
          media_type, configured_caps);
    } else {
      HOLOSCAN_LOG_INFO("Accepting any caps: {} (configured: ANY)", media_type);
    }
  } else {
    HOLOSCAN_LOG_WARN("No resource bridge available for caps validation");
  }

  /* Store caps information */
  if (holoscan_src->caps) {
    gst_caps_unref(holoscan_src->caps);
  }
  holoscan_src->caps = gst_caps_ref(caps);

  /* Mark caps as successfully negotiated */
  holoscan_src->caps_set = TRUE;
  return TRUE;
}

// Start callback
gboolean GstSrcResource::start_callback(::GstBaseSrc *src) {
  GstHoloscanSrc *holoscan_src = GST_HOLOSCAN_SRC(src);

  HOLOSCAN_LOG_INFO("Starting Holoscan bridge source");

  holoscan_src->caps_set = FALSE;

  return TRUE;
}

// Stop callback
gboolean GstSrcResource::stop_callback(::GstBaseSrc *src) {
  GstHoloscanSrc *holoscan_src = GST_HOLOSCAN_SRC(src);

  HOLOSCAN_LOG_INFO("Stopping Holoscan bridge source");

  holoscan_src->caps_set = FALSE;

  return TRUE;
}

// Create callback implementation - called by GStreamer to get the next buffer
::GstFlowReturn GstSrcResource::create_callback(::GstPushSrc *src, ::GstBuffer **buffer) {
  GstHoloscanSrc *holoscan_src = GST_HOLOSCAN_SRC(src);

  if (!holoscan_src->caps_set) {
    HOLOSCAN_LOG_ERROR("Caps not negotiated");
    return GST_FLOW_NOT_NEGOTIATED;
  }

  /* Access the GstSrcResource instance from callback */
  if (!holoscan_src->holoscan_resource) {
    HOLOSCAN_LOG_ERROR("No resource bridge available");
    return GST_FLOW_ERROR;
  }

  /* Cast back to GstSrcResource* to access C++ methods and members */
  GstSrcResource* resource = static_cast<GstSrcResource*>(holoscan_src->holoscan_resource);
  std::unique_lock<std::mutex> lock(resource->mutex_);

  /* Wait for buffer to become available or EOS */
  if (resource->buffer_queue_.empty()) {
    HOLOSCAN_LOG_DEBUG("Waiting for buffer...");
  }

  resource->queue_cv_.wait(lock, [holoscan_src]() {
    GstSrcResource* resource = static_cast<GstSrcResource*>(holoscan_src->holoscan_resource); 
    return resource == nullptr || !resource->buffer_queue_.empty();
  });
  /* Check if EOS was signaled */
  if (holoscan_src->holoscan_resource == nullptr) {
    HOLOSCAN_LOG_INFO("End of stream reached");
    return GST_FLOW_EOS;
  }

  /* Get buffer from queue */
  gst::Buffer buffer_obj = std::move(resource->buffer_queue_.front());
  resource->buffer_queue_.pop();
  
  HOLOSCAN_LOG_DEBUG("Retrieved buffer from queue, remaining: {}", resource->buffer_queue_.size());

  /* Notify waiting producers that space is available in the queue */
  resource->queue_cv_.notify_one();

  /* Transfer ownership to GStreamer (increment ref count) */
  *buffer = gst_buffer_ref(buffer_obj.get());

  return GST_FLOW_OK;
}

GstSrcResource::~GstSrcResource() {
  HOLOSCAN_LOG_INFO("Destroying GstSrcResource");

  std::unique_lock<std::mutex> lock(mutex_);
  std::queue<gst::Buffer> empty_queue;
  std::swap(buffer_queue_, empty_queue);
  GST_HOLOSCAN_SRC(src_element_future_.get().get())->holoscan_resource = nullptr;
  queue_cv_.notify_all();
  
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
  
  HOLOSCAN_LOG_INFO("Initializing GstSrcResource for data bridging");
  HOLOSCAN_LOG_INFO("Configured capabilities: '{}'", caps_.get());
  
  // Initialize GStreamer if not already done
  if (!gst_is_initialized()) {
    gst_init(nullptr, nullptr);
  }

  // Register our bridge source element type
  gst_element_register(nullptr, "holoscansrc", GST_RANK_NONE,
                      gst_holoscan_src_get_type());

  // Create the source element
  auto element = gst::make_gst_object_guard(gst_element_factory_make("holoscansrc",
                                         name().empty() ? nullptr : name().c_str()));

  if (element) {
    // Establish the bridge: set the C++ resource pointer in the C element
    GstHoloscanSrc *src = GST_HOLOSCAN_SRC(element.get());
    src->holoscan_resource = this;
    
    HOLOSCAN_LOG_INFO("GstSrcResource initialized successfully for data bridging");
    
    // Set the promise with the successfully created element
    src_element_promise_.set_value(std::move(element));
  } else {
    HOLOSCAN_LOG_ERROR("Failed to create Holoscan bridge source element");
    
    // Set exception on the promise so waiting code will be notified of failure
    src_element_promise_.set_exception(
        std::make_exception_ptr(std::runtime_error("Failed to create holoscansrc element")));
  }
}

}  // namespace holoscan

