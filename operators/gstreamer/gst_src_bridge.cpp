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

#include "gst_src_bridge.hpp"

#include <gst/app/gstappsrc.h>
#include <gst/gst.h>
#include <gst/video/video.h>

#include <chrono>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <thread>

#include <cuda_runtime_api.h>
#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
#include <gst/cuda/gstcudacontext.h>
#include <gst/cuda/gstcudamemory.h>
#ifndef GST_MAP_READ_CUDA
#define GST_MAP_READ_CUDA ((GstMapFlags)(GST_MAP_READ | GST_MAP_CUDA))
#endif
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

// Define CUDA memory feature name if not already defined by GStreamer (available since 1.22).
#ifndef GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY
#define GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY "memory:CUDAMemory"
#endif

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/logger/logger.hpp>

#include "gst/memory.hpp"
#include "gst/video_info.hpp"

namespace holoscan {

static bool gst_initialized = []() {
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
  explicit MemoryWrapper(GstVideoFormat video_format) : video_format_(video_format) {}
  virtual ~MemoryWrapper() = default;
  // Non-copyable and non-movable
  MemoryWrapper(const MemoryWrapper&) = delete;
  MemoryWrapper& operator=(const MemoryWrapper&) = delete;
  MemoryWrapper(MemoryWrapper&&) = delete;
  MemoryWrapper& operator=(MemoryWrapper&&) = delete;

  /**
   * @brief Wrap a tensor into a GStreamer memory object
   * @param tensor Tensor to wrap
   * @param user_data User data to pass to the notify callback
   * @param destroy_notify Notify callback to free the user data when the memory is destroyed
   * @return GStreamer memory object (empty on failure)
   * @throws std::runtime_error if CUDA initialization or context creation fails (CudaMemoryWrapper
   * only)
   */
  virtual gst::Memory wrap_memory(const holoscan::Tensor* tensor, void* user_data,
                                  GDestroyNotify destroy_notify) = 0;

 protected:
  GstVideoFormat video_format_;  // Video format from caps.
};

/**
 * Host memory wrapper - wraps CPU-accessible memory using standard GStreamer memory
 */
class GstSrcBridge::HostMemoryWrapper : public MemoryWrapper {
 public:
  explicit HostMemoryWrapper(GstVideoFormat video_format) : MemoryWrapper(video_format) {}

  gst::Memory wrap_memory(const holoscan::Tensor* tensor, void* user_data,
                          GDestroyNotify destroy_notify) override {
    if (!tensor->data() || tensor->nbytes() == 0) {
      HOLOSCAN_LOG_ERROR("Invalid tensor data or size for host memory wrapping");
      return gst::Memory();
    }

    HOLOSCAN_LOG_DEBUG("Wrapping as host memory (zero-copy): size={} bytes", tensor->nbytes());
    return gst::Memory(gst_memory_new_wrapped(static_cast<GstMemoryFlags>(0),  // flags
                                              tensor->data(),                  // data pointer
                                              tensor->nbytes(),                // maxsize
                                              0,                               // offset
                                              tensor->nbytes(),                // size
                                              user_data,                       // user_data
                                              destroy_notify));  // destroy_notify callback
  }
};

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
/**
 * CUDA device memory wrapper - wraps GPU memory using GStreamer CUDA memory
 * Initializes CUDA resources in constructor, throws on failure
 * Creates CUDA context on first use based on tensor's device
 */
class GstSrcBridge::CudaMemoryWrapper : public MemoryWrapper {
 public:
  explicit CudaMemoryWrapper(GstVideoFormat video_format) : MemoryWrapper(video_format) {
    HOLOSCAN_LOG_INFO("Initializing CUDA resources for zero-copy device memory");

    // Check if CUDA is available using CUDA runtime API
    cudaError_t cuda_err = cudaGetDeviceCount(&device_count_);
    if (cuda_err != cudaSuccess || device_count_ == 0) {
      std::string error_msg = fmt::format(
          "CUDA not available: {} (device count: {}). "
          "Cannot initialize CUDA resources for zero-copy device memory.",
          cudaGetErrorString(cuda_err), device_count_);
      HOLOSCAN_LOG_ERROR(error_msg);
      throw std::runtime_error(error_msg);
    }

    HOLOSCAN_LOG_INFO("CUDA detected: {} device(s) available", device_count_);

    // Initialize CUDA memory system (must be called before any GStreamer CUDA operations)
    gst_cuda_memory_init_once();

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

    HOLOSCAN_LOG_INFO("CUDA resources initialized successfully");
  }

  gst::Memory wrap_memory(const holoscan::Tensor* tensor, void* user_data,
                          GDestroyNotify destroy_notify) override {
    void* tensor_data = tensor->data();
    size_t tensor_size = tensor->nbytes();

    if (!tensor_data || tensor_size == 0) {
      HOLOSCAN_LOG_ERROR("Invalid tensor data or size for CUDA memory wrapping");
      return gst::Memory();
    }

    // Detect CUDA device from tensor
    gint cuda_device_id = tensor->device().device_id;

    // Validate device ID
    if (cuda_device_id < 0 || cuda_device_id >= device_count_) {
      HOLOSCAN_LOG_ERROR("Invalid CUDA device ID {} from tensor (available: 0-{})", cuda_device_id,
                         device_count_ - 1);
      return gst::Memory();
    }

    if (!cuda_context_ || current_device_id_ != cuda_device_id) {
      if (current_device_id_ >= 0 && current_device_id_ != cuda_device_id) {
        HOLOSCAN_LOG_ERROR(
            "Cannot switch CUDA devices within the same encoder. Current: {}, Requested: {}",
            current_device_id_, cuda_device_id);
        return gst::Memory();
      }

      HOLOSCAN_LOG_INFO("Creating CUDA context for device {}", cuda_device_id);
      cuda_context_ = gst::CudaContext(gst_cuda_context_new(cuda_device_id));
      if (!cuda_context_) {
        HOLOSCAN_LOG_ERROR("Failed to create CUDA context for device {}", cuda_device_id);
        throw std::runtime_error("Failed to create CUDA context for device " +
                                 std::to_string(cuda_device_id));
      }
      current_device_id_ = cuda_device_id;
      HOLOSCAN_LOG_INFO("CUDA context created for device {}", cuda_device_id);
    }

    // Get tensor shape for width and height
    auto shape = tensor->shape();
    if (shape.size() < 2) {
      HOLOSCAN_LOG_ERROR("Tensor has invalid rank {} for CUDA wrapping", shape.size());
      return gst::Memory();
    }

    // Tensor is in format: [height, width, channels] or [height, width].
    gint height = shape[0];
    gint width = shape[1];

    HOLOSCAN_LOG_DEBUG("Wrapping tensor with format {} ({}x{})",
                       gst_video_format_to_string(video_format_), width, height);

    // Create video info for the tensor using format from caps.
    GstVideoInfo video_info;
    gst_video_info_init(&video_info);
    if (!gst_video_info_set_format(&video_info, video_format_, width, height)) {
      HOLOSCAN_LOG_ERROR("Failed to set video info for CUDA memory wrapping");
      return gst::Memory();
    }

    HOLOSCAN_LOG_DEBUG("Wrapping as CUDA device memory (zero-copy): device={}, size={} bytes",
                       cuda_device_id, tensor_size);

    // Wrap the device memory pointer in GstCudaMemory
    return gst::Memory(gst_cuda_allocator_alloc_wrapped(
        GST_CUDA_ALLOCATOR(cuda_allocator_.get()), cuda_context_.get(),
        nullptr,                                     // CUDA stream (nullptr = default)
        &video_info,                                 // video info
        reinterpret_cast<CUdeviceptr>(tensor_data),  // device pointer
        user_data,                                   // user_data
        destroy_notify));                            // destroy_notify callback
  }

 private:
  int device_count_ = 0;
  gint current_device_id_ = -1;  // -1 means no context created yet
  gst::CudaContext cuda_context_;
  gst::Allocator cuda_allocator_;
};
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

namespace {

// ============================================================================
// Tensor Wrapper for Memory Lifetime Management
// ============================================================================

// Wrapper to keep tensor alive while GStreamer uses its memory
struct TensorWrapper {
  explicit TensorWrapper(std::shared_ptr<holoscan::DLManagedTensorContext> ctx)
      : dl_ctx(std::move(ctx)) {}

  std::shared_ptr<holoscan::DLManagedTensorContext> dl_ctx;  // Keep tensor memory alive
};

// Callback to free TensorWrapper when GstMemory is destroyed
void free_tensor_wrapper(gpointer user_data) {
  auto* wrapper = static_cast<TensorWrapper*>(user_data);
  delete wrapper;
}

/**
 * @brief Create memory wrapper based on tensor storage type and caps
 * @param tensor Tensor to inspect for storage type
 * @param caps Capabilities string to check for CUDA memory request
 * @return Shared pointer to the appropriate memory wrapper
 */
std::shared_ptr<GstSrcBridge::MemoryWrapper> create_memory_wrapper(const holoscan::Tensor* tensor,
                                                                   const std::string& caps) {
  // Extract video format from caps.
  GstVideoFormat video_format = GST_VIDEO_FORMAT_UNKNOWN;
  gst::Caps gst_caps(caps);
  if (!gst_caps) {
    HOLOSCAN_LOG_ERROR("Failed to parse caps string");
    return nullptr;
  }
  video_format = gst_caps.get_video_info()->get_format();

  // Check if CUDA memory is requested in caps using proper GStreamer API.
  bool cuda_requested = gst_caps.has_feature(GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY);

  // Check device type from DLPack
  DLDevice device = tensor->device();
  bool is_device_memory = (device.device_type == kDLCUDA || device.device_type == kDLCUDAManaged);
  auto storage_type_str = is_device_memory ? "GPU" : "CPU";

  // Use CudaMemoryWrapper for GPU tensors when GStreamer requests CUDA memory
  if (is_device_memory && cuda_requested) {
#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
    HOLOSCAN_LOG_INFO("Creating CUDA memory wrapper ({} memory)", storage_type_str);
    return std::make_shared<GstSrcBridge::CudaMemoryWrapper>(video_format);
#else
    HOLOSCAN_LOG_WARN(
        "CUDA memory requested but not available (requires GStreamer 1.24+). Falling back to host "
        "memory.");
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT
  }
  // Use HostMemoryWrapper for CPU memory or when CUDA not requested
  HOLOSCAN_LOG_INFO("Creating host memory wrapper ({} memory)", storage_type_str);
  return std::make_shared<GstSrcBridge::HostMemoryWrapper>(video_format);
}

}  // anonymous namespace

// ============================================================================
// GstSrcBridge Implementation
// ============================================================================

GstSrcBridge::GstSrcBridge(const std::string& name, const std::string& caps_string,
                           size_t max_buffers)
    : name_(name),
      caps_string_(caps_string),
      max_buffers_(max_buffers),
      src_element_(gst_element_factory_make("appsrc", name_.empty() ? nullptr : name_.c_str())) {
  HOLOSCAN_LOG_INFO("Creating GstSrcBridge: name='{}', caps='{}', max_buffers={}", name_,
                    caps_string_, max_buffers_);

  if (!src_element_) {
    HOLOSCAN_LOG_ERROR("Failed to create appsrc element");
    throw std::runtime_error("Failed to create appsrc element");
  }

  gst::Caps caps(caps_string_);
  if (!caps) {
    HOLOSCAN_LOG_ERROR("Failed to parse configured caps: '{}'", caps_string_);
    throw std::runtime_error("Failed to parse caps");
  }

  HOLOSCAN_LOG_INFO("Set appsrc caps: {}", caps.to_string());

  // Parse framerate from caps first to determine is-live mode
  bool is_live = false;
  // Extract framerate from caps
  if (caps.get_size() > 0) {
    const GValue* framerate_value = caps.get_structure_value("framerate");

    if (framerate_value && GST_VALUE_HOLDS_FRACTION(framerate_value)) {
      framerate_num_ = gst_value_get_fraction_numerator(framerate_value);
      framerate_den_ = gst_value_get_fraction_denominator(framerate_value);

      // If framerate is 0, treat as live source (process frames as fast as they come)
      if (framerate_num_ == 0) {
        is_live = true;
        HOLOSCAN_LOG_INFO("Framerate is 0/1 - using live mode (no framerate control)");
      } else {
        HOLOSCAN_LOG_INFO("Parsed framerate from caps: {}/{} fps", framerate_num_, framerate_den_);
      }
    } else {
      // No framerate specified - use live mode
      is_live = true;
      framerate_num_ = 0;
      framerate_den_ = 1;
      HOLOSCAN_LOG_INFO("Framerate not found in caps - using live mode (no framerate control)");
    }
  }

  // Configure appsrc properties
  GstAppSrc* appsrc = GST_APP_SRC(src_element_.get());
  g_object_set(appsrc, "stream-type", GST_APP_STREAM_TYPE_STREAM,  // Continuous stream
               "format", GST_FORMAT_TIME,                          // Time-based format
               "is-live", is_live ? TRUE : FALSE,  // Live mode if framerate is 0 or not specified
               "max-buffers", max_buffers_,        // Buffer queue limit (0 = unlimited)
               "max-bytes", (guint64)0,  // Byte limit (0 = unlimited, controlled by max-buffers)
               "block", TRUE,  // Block push_buffer() when queue is full for proper flow control
               NULL);

  // Set caps
  gst_app_src_set_caps(appsrc, caps.get());

  HOLOSCAN_LOG_INFO("GstSrcBridge initialized with appsrc (max_buffers: {}, framerate: {}/{})",
                    max_buffers_, framerate_num_, framerate_den_);
}

GstSrcBridge::~GstSrcBridge() { HOLOSCAN_LOG_INFO("Destroying GstSrcBridge"); }

gst::Element GstSrcBridge::get_gst_element() const { return src_element_; }

bool GstSrcBridge::push_buffer(gst::Buffer buffer) {
  HOLOSCAN_LOG_DEBUG("GstSrcBridge::push_buffer() - Starting");

  if (!buffer) {
    HOLOSCAN_LOG_ERROR("Invalid buffer provided to push_buffer");
    return false;
  }

  GstAppSrc* appsrc = GST_APP_SRC(src_element_.get());

  // Push the buffer to appsrc (transfers ownership)
  // Note: gst_app_src_push_buffer takes ownership and will unref the buffer
  // We use ref() to give appsrc its own reference while we keep ours
  GstFlowReturn ret = gst_app_src_push_buffer(appsrc, buffer.ref());
  switch (ret) {
    case GST_FLOW_OK:
      HOLOSCAN_LOG_DEBUG("Successfully pushed buffer to appsrc");
      break;
    case GST_FLOW_FLUSHING:
      HOLOSCAN_LOG_WARN("appsrc is flushing, buffer not pushed");
      break;
    case GST_FLOW_EOS:
      HOLOSCAN_LOG_WARN("appsrc is in EOS state");
      break;
    default:
      HOLOSCAN_LOG_ERROR("Failed to push buffer to appsrc: {}", gst_flow_get_name(ret));
      break;
  }
  return ret == GST_FLOW_OK;
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
  gst::Pad pad(gst_element_get_static_pad(src_element_.get(), "src"));
  if (!pad) return gst::Caps();  // Return empty caps

  return gst::Caps(gst_pad_get_current_caps(pad.get()));
}

gst::Buffer GstSrcBridge::create_buffer_from_tensor_map(const TensorMap& tensor_map) {
  // Create an empty GStreamer buffer at the start
  gst::Buffer buffer(gst_buffer_new());

  if (tensor_map.empty()) {
    HOLOSCAN_LOG_ERROR("TensorMap is empty");
    return buffer;
  }

  int tensor_count = 0;
  size_t total_size = 0;

  // Iterate through all tensors in the map
  for (const auto& [name, tensor_ptr] : tensor_map) {
    if (!tensor_ptr) {
      HOLOSCAN_LOG_WARN("Skipping null tensor '{}'", name);
      continue;
    }

    // Get tensor data
    size_t tensor_size = tensor_ptr->nbytes();
    void* tensor_data = tensor_ptr->data();

    if (!tensor_data || tensor_size == 0) {
      HOLOSCAN_LOG_WARN("Skipping tensor '{}' - invalid data or size", name);
      continue;
    }

    // Lazy initialization of memory wrapper on first tensor
    if (!memory_wrapper_) {
      try {
        memory_wrapper_ = create_memory_wrapper(tensor_ptr.get(), caps_string_);
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Failed to create memory wrapper: {}", e.what());
        return buffer;
      }
    }

    // Create a TensorWrapper with the shared_ptr to keep tensor memory alive
    auto tensor_wrapper = std::make_unique<TensorWrapper>(tensor_ptr->dl_ctx());

    // Use the memory wrapper to wrap the tensor
    auto memory =
        memory_wrapper_->wrap_memory(tensor_ptr.get(), tensor_wrapper.get(), free_tensor_wrapper);

    if (!memory) {
      HOLOSCAN_LOG_ERROR("Failed to wrap memory for tensor '{}'", name);
      continue;
    }

    // Release ownership - GStreamer now manages the wrapper lifetime
    tensor_wrapper.release();

    // Append wrapped memory to buffer
    buffer.append_memory(memory);

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
      timestamp = g_get_monotonic_time() * 1000;  // Convert microseconds to nanoseconds
      duration = GST_CLOCK_TIME_NONE;             // Duration unknown in live mode
    } else {
      // Calculate timestamp directly from frame count to avoid accumulating rounding errors
      timestamp = gst_util_uint64_scale(frame_count_, framerate_den_ * GST_SECOND, framerate_num_);

      // Calculate duration as difference between next and current timestamp
      GstClockTime next_timestamp =
          gst_util_uint64_scale(frame_count_ + 1, framerate_den_ * GST_SECOND, framerate_num_);
      duration = next_timestamp - timestamp;
    }

    buffer->pts = timestamp;
    buffer->dts = timestamp;
    buffer->duration = duration;
    frame_count_++;

    HOLOSCAN_LOG_DEBUG(
        "Successfully created zero-copy GStreamer buffer: {} tensors, {} total bytes, frame={}, "
        "PTS={}",
        tensor_count, total_size, frame_count_ - 1,
        (timestamp == GST_CLOCK_TIME_NONE) ? "NONE" : std::to_string(timestamp) + " ns");
  }

  return buffer;
}

}  // namespace holoscan
