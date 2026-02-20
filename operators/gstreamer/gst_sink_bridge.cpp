/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gst_sink_bridge.hpp"

#include <gst/gst.h>
#include <gst/video/video.h>

#include <array>

#include <cuda_runtime_api.h>

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
#include <gst/cuda/gstcudamemory.h>
#ifndef GST_MAP_READ_CUDA
#define GST_MAP_READ_CUDA (static_cast<::GstMapFlags>(GST_MAP_READ | GST_MAP_CUDA))
#endif
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

// Define CUDA memory feature name if not already defined by GStreamer (available since 1.22).
#ifndef GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY
#define GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY "memory:CUDAMemory"
#endif

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/logger/logger.hpp>

#include "gst/config.hpp"
#include "gst/memory.hpp"
#include "gst/sample.hpp"
#include "gst/video_info.hpp"

namespace holoscan {

static bool gst_initialized = []() {
  // Initialize GStreamer if not already done
  if (!gst_is_initialized()) {
    gst_init(nullptr, nullptr);
  }
  return true;
}();

namespace {

// ============================================================================
// Memory Mapper Classes
// ============================================================================

/**
 * @brief Abstract base class for mapping GStreamer memory to tensor data
 */
class MemoryMapper {
 public:
  virtual ~MemoryMapper() = default;

  // Non-copyable and non-movable
  MemoryMapper(const MemoryMapper&) = delete;
  MemoryMapper& operator=(const MemoryMapper&) = delete;
  MemoryMapper(MemoryMapper&&) = delete;
  MemoryMapper& operator=(MemoryMapper&&) = delete;

  /**
   * @brief Result of mapping a GStreamer memory block
   */
  struct MappedMemory {
    ::GstMapInfo map_info{};
    DLDeviceType device_type = kDLCPU;
    nvidia::gxf::MemoryStorageType storage_type = nvidia::gxf::MemoryStorageType::kSystem;
  };

  /**
   * @brief Map GStreamer memory for reading
   * @param memory GStreamer memory block to map
   * @return MappedMemory if successful, std::nullopt on failure
   */
  virtual std::optional<MappedMemory> map_memory(gst::Memory& memory) = 0;

 protected:
  MemoryMapper() = default;

  /**
   * @brief Helper to map GStreamer memory with the specified flags
   *
   * Maps the memory and detects the actual device type using CUDA API.
   * Returns MappedMemory with map_info, device_type and storage_type populated.
   * Data pointer and size can be obtained from map_info.data and map_info.size.
   *
   * @param memory GStreamer memory block to map
   * @param flags Mapping flags (GST_MAP_READ, GST_MAP_READ_CUDA, etc.)
   * @return MappedMemory if mapping succeeded, std::nullopt otherwise
   */
  std::optional<MappedMemory> map_memory_internal(const gst::Memory& memory, ::GstMapFlags flags) {
    MappedMemory result;

    if (!memory.map(&result.map_info, flags)) {
      return std::nullopt;
    }

    // Use CUDA API to accurately detect the memory type
    cudaPointerAttributes attributes;
    cudaError_t cuda_result = cudaPointerGetAttributes(&attributes, result.map_info.data);

    if (cuda_result != cudaSuccess) {
      cudaGetLastError();  // Reset error
      result.device_type = kDLCPU;
    } else {
      // Convert CUDA memory type to DLDeviceType
      switch (attributes.type) {
        case cudaMemoryTypeDevice:
          result.device_type = kDLCUDA;
          break;
        case cudaMemoryTypeHost:
          result.device_type = kDLCUDAHost;
          break;
        case cudaMemoryTypeManaged:
          result.device_type = kDLCUDAManaged;
          break;
        default:
          result.device_type = kDLCPU;
      }
    }

    // Set storage type based on actual device type
    if (result.device_type == kDLCUDA || result.device_type == kDLCUDAManaged) {
      result.storage_type = nvidia::gxf::MemoryStorageType::kDevice;
    } else if (result.device_type == kDLCUDAHost) {
      result.storage_type = nvidia::gxf::MemoryStorageType::kHost;
    } else {
      result.storage_type = nvidia::gxf::MemoryStorageType::kSystem;
    }

    return result;
  }
};

/**
 * @brief Host (CPU) memory mapper - maps system memory for CPU access
 */
class HostMemoryMapper : public MemoryMapper {
 public:
  HostMemoryMapper() = default;

  std::optional<MappedMemory> map_memory(gst::Memory& memory) override {
    auto result = map_memory_internal(memory, GST_MAP_READ);
    if (!result) {
      HOLOSCAN_LOG_ERROR("Failed to map host memory");
      return std::nullopt;
    }

    HOLOSCAN_LOG_DEBUG("Mapped as host memory: {} bytes", result->map_info.size);
    return result;
  }
};

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
/**
 * @brief CUDA device memory mapper - maps GPU memory for device access
 *
 * Attempts to map as CUDA memory first (GST_MAP_READ_CUDA), then falls back
 * to host memory mapping if that fails by delegating to parent class.
 */
class CudaMemoryMapper : public HostMemoryMapper {
 public:
  CudaMemoryMapper() = default;

  std::optional<MappedMemory> map_memory(gst::Memory& memory) override {
    // Try CUDA mapping first
    auto result = map_memory_internal(memory, GST_MAP_READ_CUDA);
    if (result) {
      HOLOSCAN_LOG_DEBUG("Mapped as CUDA memory: {} bytes", result->map_info.size);
      return result;
    }

    // Fall back to parent's host memory mapping
    HOLOSCAN_LOG_DEBUG("CUDA mapping failed, falling back to host memory mapping");
    return HostMemoryMapper::map_memory(memory);
  }
};
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

/**
 * @brief Create appropriate memory mapper based on caps
 * @param caps Capabilities to check for CUDA memory feature
 * @return Shared pointer to the appropriate memory mapper
 */
std::shared_ptr<MemoryMapper> create_memory_mapper(const gst::Caps& caps) {
  // Check if CUDA memory is indicated in caps
  bool cuda_requested = caps.has_feature(GST_CAPS_FEATURE_MEMORY_CUDA_MEMORY);

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
  if (cuda_requested) {
    HOLOSCAN_LOG_INFO("Creating CUDA memory mapper for sink");
    return std::make_shared<CudaMemoryMapper>();
  }
#else
  if (cuda_requested) {
    HOLOSCAN_LOG_WARN(
        "CUDA memory requested in caps, but built without CUDA support. Using host mapper.");
  }
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

  HOLOSCAN_LOG_INFO("Creating host memory mapper for sink");
  return std::make_shared<HostMemoryMapper>();
}

}  // namespace

// ============================================================================
// Tensor Builder Classes
// ============================================================================

/**
 * @brief Abstract base class for building tensors from mapped memory
 *
 * Encapsulates all logic for creating tensors from GStreamer memory blocks,
 * including naming, shape calculation, and stride handling. This is a nested
 * class because it's referenced in GstSinkBridge's public interface.
 */
class GstSinkBridge::TensorBuilder {
 public:
  virtual ~TensorBuilder() = default;

  // Non-copyable and non-movable
  TensorBuilder(const TensorBuilder&) = delete;
  TensorBuilder& operator=(const TensorBuilder&) = delete;
  TensorBuilder(TensorBuilder&&) = delete;
  TensorBuilder& operator=(TensorBuilder&&) = delete;

  /**
   * @brief Create a complete tensor with its name from a GStreamer buffer
   *
   * Template Method: Extracts memory, maps it, calls virtual function for metadata,
   * wraps memory, and converts to Holoscan tensor. Common logic is handled here,
   * while derived classes provide specific metadata via compute_tensor_metadata().
   *
   * @param buffer GStreamer buffer containing the memory (captured for lifetime management)
   * @param mem_idx Index of the memory block to extract from the buffer
   * @param mem_count Total number of memory blocks in the buffer
   * @return Pair of (tensor_name, tensor_ptr). Returns empty string and nullptr on failure.
   */
  std::pair<std::string, std::shared_ptr<holoscan::Tensor>> create_tensor(
      gst::Buffer buffer,
      guint mem_idx,
      guint mem_count) {
    // Extract memory block from buffer
    gst::Memory memory(buffer.get_memory(mem_idx));
    if (!memory) {
      HOLOSCAN_LOG_ERROR("No memory found for block {}", mem_idx);
      return {"", nullptr};
    }

    // Map memory
    auto mapped = memory_mapper_->map_memory(memory);
    if (!mapped) {
      HOLOSCAN_LOG_ERROR("Failed to map memory for block {}", mem_idx);
      return {"", nullptr};
    }

    // Call virtual function to compute tensor-specific metadata
    std::string tensor_name;
    nvidia::gxf::Shape shape;
    std::array<size_t, 8> strides;

    if (!compute_tensor_metadata(memory, *mapped, mem_idx, mem_count,
                                  tensor_name, shape, strides)) {
      HOLOSCAN_LOG_ERROR("Failed to compute tensor metadata for block {}", mem_idx);
      return {"", nullptr};
    }

    // Create GXF tensor and wrap GStreamer memory with custom deleter
    auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();

    // Wrap memory with GXF tensor (handles lifetime management)
    nvidia::gxf::PrimitiveType primitive_type = nvidia::gxf::PrimitiveType::kUnsigned8;
    uint64_t element_size = nvidia::gxf::PrimitiveTypeSize(primitive_type);

    std::function<nvidia::gxf::Expected<void>(void*)> deleter =
        [buffer = std::move(buffer), memory, map_info = mapped->map_info](void*) mutable {
      memory.unmap(const_cast<::GstMapInfo*>(&map_info));
      return nvidia::gxf::Success;
    };

    gxf_tensor->wrapMemory(shape, primitive_type, element_size, strides,
                           mapped->storage_type, static_cast<uint8_t*>(mapped->map_info.data),
                           deleter);

    // Convert GXF tensor to Holoscan tensor via DLPack
    auto maybe_dl_ctx = gxf_tensor->toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
      HOLOSCAN_LOG_ERROR("Failed to convert GXF tensor to Holoscan tensor for block {}", mem_idx);
      return {"", nullptr};
    }

    return {std::move(tensor_name), std::make_shared<holoscan::Tensor>(maybe_dl_ctx.value())};
  }

 protected:
  explicit TensorBuilder(const gst::Caps& caps)
      : memory_mapper_(create_memory_mapper(caps)) {}

  // Memory mapper owned by this builder
  std::shared_ptr<MemoryMapper> memory_mapper_;

 private:
  /**
   * @brief Compute tensor-specific metadata (name, shape, strides)
   *
   * Private virtual function called by create_tensor() to compute tensor-specific
   * metadata. Derived classes implement this to provide their unique logic.
   * Only the base class template method should call this.
   *
   * @param memory GStreamer memory object (for size queries, CUDA casting, etc.)
   * @param mapped Mapped memory information (device type, storage type, etc.)
   * @param mem_idx Index of the memory block in the buffer
   * @param mem_count Total number of memory blocks in the buffer
   * @param[out] tensor_name Output: name for the tensor (e.g., "video_frame", "data")
   * @param[out] shape Output: tensor shape (dimensions)
   * @param[out] strides Output: tensor strides (byte offsets between elements)
   * @return true if metadata computed successfully, false on failure
   */
  virtual bool compute_tensor_metadata(
      gst::Memory& memory,
      const MemoryMapper::MappedMemory& mapped,
      guint mem_idx,
      guint mem_count,
      std::string& tensor_name,
      nvidia::gxf::Shape& shape,
      std::array<size_t, 8>& strides) = 0;
};

namespace {

/**
 * @brief Video tensor builder - creates tensors from video plane data
 *
 * Handles video-specific logic like plane naming, multi-dimensional shapes,
 * and CUDA stride overrides for GPU memory alignment.
 */
class VideoTensorBuilder : public GstSinkBridge::TensorBuilder {
 public:
  VideoTensorBuilder(const gst::Caps& caps, gst::VideoInfo video_info)
      : TensorBuilder(caps), video_info_(std::move(video_info)) {}

 private:
  bool compute_tensor_metadata(
      gst::Memory& memory,
      const MemoryMapper::MappedMemory& mapped,
      guint mem_idx,
      guint mem_count,
      std::string& tensor_name,
      nvidia::gxf::Shape& shape,
      std::array<size_t, 8>& strides) override {
    // Generate tensor name with appropriate suffix
    guint n_planes = video_info_.get_n_planes();
    static constexpr std::array<const char*, 4> plane_suffixes = {"", "_u", "_v", "_a"};
    tensor_name = "video_frame";

    if (n_planes > 1) {
      if (n_planes == 2 && mem_idx == 1) {
        tensor_name += "_uv";  // NV12 format
      } else if (mem_idx > 0 && mem_idx < plane_suffixes.size()) {
        tensor_name += plane_suffixes[mem_idx];  // I420 format
      }
    }

    // Get plane-specific dimensions
    int plane_width = video_info_.get_comp_width(mem_idx);
    int plane_height = video_info_.get_comp_height(mem_idx);
    int plane_stride = video_info_.get_stride(mem_idx);

    // For packed formats (single plane), use component count and pixel stride
    // For planar formats, each plane is treated as single-component
    guint plane_components = (mem_idx == 0 && n_planes == 1) ? video_info_.get_n_components() : 1;
    size_t bytes_per_pixel = (mem_idx == 0 && n_planes == 1) ? video_info_.get_comp_pstride(0) : 1;

    // Create shape: [height, width, components]
    shape = nvidia::gxf::Shape({static_cast<int32_t>(plane_height),
                                static_cast<int32_t>(plane_width),
                                static_cast<int32_t>(plane_components)});

    // Calculate strides: [row_stride, bytes_per_pixel, bytes_per_component]
    strides = {{static_cast<size_t>(plane_stride), bytes_per_pixel, sizeof(uint8_t),
                0, 0, 0, 0, 0}};

#if HOLOSCAN_GSTREAMER_CUDA_SUPPORT
    // For CUDA memory, use the actual stride from GstCudaMemory
    // CUDA memory may have different pitch due to alignment requirements
    if (mapped.device_type == kDLCUDA || mapped.device_type == kDLCUDAManaged) {
      ::GstCudaMemory* cuda_mem = GST_CUDA_MEMORY_CAST(memory.get());
      if (cuda_mem && cuda_mem->info.stride[0] > 0) {
        size_t old_stride = strides[0];
        strides[0] = cuda_mem->info.stride[0];
        HOLOSCAN_LOG_DEBUG("Applied CUDA stride: {} bytes (VideoInfo reported: {} bytes)",
                          cuda_mem->info.stride[0], old_stride);
      }
    }
#endif  // HOLOSCAN_GSTREAMER_CUDA_SUPPORT

    return true;
  }

  gst::VideoInfo video_info_;
};

/**
 * @brief Generic tensor builder - creates 1D byte array tensors
 *
 * Used for non-video formats or when caps don't contain video info.
 * Creates simple flat byte array tensors.
 */
class GenericTensorBuilder : public GstSinkBridge::TensorBuilder {
 public:
  explicit GenericTensorBuilder(const gst::Caps& caps)
      : TensorBuilder(caps) {}

 private:
  bool compute_tensor_metadata(
      gst::Memory& memory,
      const MemoryMapper::MappedMemory& mapped,
      guint mem_idx,
      guint mem_count,
      std::string& tensor_name,
      nvidia::gxf::Shape& shape,
      std::array<size_t, 8>& strides) override {
    // Generate tensor name
    tensor_name = (mem_count == 1) ? "data" : fmt::format("data_{}", mem_idx);

    // Create shape: 1D byte array
    gsize size = memory.get_sizes();
    shape = nvidia::gxf::Shape({static_cast<int32_t>(size)});

    // Strides: just element size
    strides = {{1, 0, 0, 0, 0, 0, 0, 0}};

    return true;
  }
};

/**
 * @brief Create appropriate tensor builder based on caps
 * @param caps Capabilities to check for video format and memory type
 * @return Shared pointer to the appropriate tensor builder (creates memory mapper internally)
 */
std::shared_ptr<GstSinkBridge::TensorBuilder> create_tensor_builder(const gst::Caps& caps) {
  // Try to extract video info from caps
  auto video_info_opt = caps.get_video_info();

  if (video_info_opt.has_value()) {
    HOLOSCAN_LOG_INFO("Creating video tensor builder for sink");
    return std::make_shared<VideoTensorBuilder>(caps, *video_info_opt);
  }

  HOLOSCAN_LOG_INFO("Creating generic tensor builder for sink");
  return std::make_shared<GenericTensorBuilder>(caps);
}

}  // namespace

// ============================================================================
// GstSinkBridge Implementation
// ============================================================================

GstSinkBridge::GstSinkBridge(const std::string& name, const std::string& caps_string,
                             size_t max_buffers, bool qos)
    : name_(name),
      sink_element_(gst::static_object_cast<gst::AppSink>(
          gst::Element(gst_element_factory_make("appsink", name_.empty() ? nullptr : name_.c_str()))
              .ref_sink())),
      eos_future_(eos_promise_.get_future()) {
  HOLOSCAN_LOG_INFO("Creating GstSinkBridge: name='{}', caps='{}', max_buffers={}, qos={}", name_,
                    caps_string, max_buffers, qos ? "enabled" : "disabled");

  if (!sink_element_) {
    HOLOSCAN_LOG_ERROR("Failed to create appsink element");
    throw std::runtime_error("Failed to create appsink element");
  }

  gst::Caps caps(caps_string);
  if (!caps) {
    HOLOSCAN_LOG_ERROR("Failed to parse configured caps: '{}'", caps_string);
    throw std::runtime_error("Failed to parse caps");
  }

  // Configure appsink properties
  sink_element_.set_properties(
      "emit-signals", false,  // Use callbacks instead of signals (more efficient)
      "sync", true,  // Sync to clock for proper timing
      "max-buffers", static_cast<guint>(max_buffers),  // Buffer queue limit
      "drop", false,  // Don't drop buffers - we handle backpressure
                      // Note: Deprecated in GStreamer ≥1.28, replace with "leaky-type"
      "qos", qos,  // Enable/disable Quality of Service (frame dropping)
      "caps", caps);  // Capabilities for the appsink (automatically unwrapped)

  // Set up appsink callbacks (EOS, preroll, sample).
  // We use pure pull-on-demand: worker pulls only when consumer requests.
  // This relies entirely on GStreamer's built-in backpressure via max-buffers.
  ::GstAppSinkCallbacks callbacks = {};  // Zero-initialize
  callbacks.eos = eos_callback;
  callbacks.new_preroll = new_preroll_callback;
  callbacks.new_sample = new_sample_callback;

  // Attach callbacks with 'this' as user_data
  sink_element_.set_callbacks(&callbacks, this, NULL);

  HOLOSCAN_LOG_INFO("GstSinkBridge initialized with appsink (max_buffers: {}, QoS: {})",
                    max_buffers, qos ? "enabled" : "disabled");
}

GstSinkBridge::~GstSinkBridge() {
  HOLOSCAN_LOG_INFO("Destroying GstSinkBridge");

  // Unregister callbacks to prevent NEW callbacks from being dispatched.
  // Note: GStreamer requires a valid (but zeroed) structure, not NULL.
  ::GstAppSinkCallbacks empty_callbacks = {};  // Zero-initialize
  gst_app_sink_set_callbacks(GST_APP_SINK(sink_element_.get()), &empty_callbacks, NULL, NULL);

  // Wait for any in-flight EOS callbacks to complete.
  // This prevents destruction while any operation is still accessing our members.
  active_operations_.wait();

  HOLOSCAN_LOG_INFO("GstSinkBridge ready to be destroyed");
}

gst::Element GstSinkBridge::get_gst_element() const {
  // Use static_object_cast for type-safe casting from AppSink to Element
  return gst::static_object_cast<gst::Element>(sink_element_);
}

std::future<gst::Buffer> GstSinkBridge::pull_buffer() {
  // Create a promise for this request.
  std::promise<gst::Buffer> promise;
  auto future = promise.get_future();

  std::lock_guard<std::mutex> lock(queue_mutex_);

  // Queue the promise first to avoid race conditions where a sample becomes
  // available between checking and queuing.
  pending_requests_.push(std::move(promise));

  // Check if a sample is already available in the appsink queue.
  gst::Sample sample = sink_element_.try_pull_sample(0);
  if (sample) {
    // Sample available - fulfill the oldest pending request (FIFO order, maintains fairness).
    std::promise<gst::Buffer> queued_promise = std::move(pending_requests_.front());
    pending_requests_.pop();
    queued_promise.set_value(sample.get_buffer());
    HOLOSCAN_LOG_DEBUG("Fulfilled oldest pending request with available sample");
  } else {
    // No buffers available, promise stays queued for callback fulfillment.
    HOLOSCAN_LOG_DEBUG("Queued pending buffer request, total pending: {}",
                       pending_requests_.size());
  }

  return future;
}

std::shared_future<void> GstSinkBridge::get_eos_future() const { return eos_future_; }

gst::Caps GstSinkBridge::get_current_caps() const {
  gst::Pad pad = sink_element_.get_static_pad("sink");
  if (!pad)
    return gst::Caps();  // Return empty caps

  return pad.get_current_caps();
}

TensorMap GstSinkBridge::create_tensor_map_from_buffer(gst::Buffer buffer) const {
  // Validate buffer
  if (!buffer) {
    HOLOSCAN_LOG_ERROR("Invalid/null buffer provided");
    return TensorMap();
  }

  // Get current negotiated caps
  const gst::Caps caps = get_current_caps();

  // Create TensorMap to hold tensor(s)
  TensorMap tensor_map;

  // Validate caps
  if (caps.is_empty()) {
    HOLOSCAN_LOG_ERROR("No caps available for buffer");
    return tensor_map;
  }

  // Lazy initialization of tensor builder (which owns a memory mapper) based on caps
  if (!tensor_builder_) {
    tensor_builder_ = create_tensor_builder(caps);
  }

  // Get number of memory blocks in the buffer
  guint mem_count = buffer.n_memory();

  // Process each memory block
  for (guint mem_idx = 0; mem_idx < mem_count; mem_idx++) {
    // Create tensor with name using polymorphic builder
    // (extracts memory, maps it, and creates tensor)
    auto [tensor_name, tensor] = tensor_builder_->create_tensor(buffer, mem_idx, mem_count);
    if (!tensor) {
      HOLOSCAN_LOG_ERROR("Failed to create tensor '{}'",
                         tensor_name.empty() ? "<unnamed>" : tensor_name);
      return TensorMap();
    }

    // Add tensor to map
    tensor_map[tensor_name] = tensor;
  }
  return tensor_map;
}
// ============================================================================
// Callback functions for appsink
// ============================================================================

void GstSinkBridge::eos_callback(::GstAppSink* /*appsink*/, ::gpointer user_data) {
  GstSinkBridge* bridge = static_cast<GstSinkBridge*>(user_data);

  // RAII guard to track this callback and ensure wait group is released on exit.
  GstWaitGroupGuard guard(bridge->active_operations_);

  HOLOSCAN_LOG_INFO("EOS event received in appsink: {}", bridge->name_);

  // Set the EOS promise to signal that all data has been processed.
  // Make this idempotent - EOS can be delivered multiple times during teardown/state transitions.
  bool expected = false;
  if (bridge->eos_signaled_.compare_exchange_strong(expected, true)) {
    bridge->eos_promise_.set_value();
  } else {
    HOLOSCAN_LOG_DEBUG("EOS already signaled, ignoring duplicate");
  }
}

::GstFlowReturn GstSinkBridge::handle_buffer_common(::GstAppSink* appsink,
                                                    ::GstSample* (*pull_func)(::GstAppSink*),
                                                    const char* callback_name) {
  // RAII guard to track this callback and ensure wait group is released on exit.
  GstWaitGroupGuard guard(active_operations_);
  HOLOSCAN_LOG_DEBUG("Received buffer via {}", callback_name);
  // Lock for thread-safe queue access.
  std::unique_lock<std::mutex> lock(queue_mutex_);

  // Check if there are pending requests waiting for buffers.
  if (!pending_requests_.empty()) {
    gst::Sample sample(pull_func(appsink));
    if (!sample) {
      HOLOSCAN_LOG_ERROR("Failed to pull sample from appsink ({})", callback_name);
      return GST_FLOW_ERROR;
    }

    // Extract buffer from sample (RAII wrapper manages lifetime).
    gst::Buffer buffer(sample.get_buffer());
    if (!buffer) {
      HOLOSCAN_LOG_ERROR("No buffer in sample ({})", callback_name);
      return GST_FLOW_ERROR;
    }

    // Fulfill the oldest pending request (FIFO order).
    std::promise<gst::Buffer> promise = std::move(pending_requests_.front());
    pending_requests_.pop();
    promise.set_value(std::move(buffer));
    HOLOSCAN_LOG_DEBUG("Fulfilled pending buffer request, remaining pending: {} ({})",
                       pending_requests_.size(), callback_name);
  }

  return GST_FLOW_OK;
}

::GstFlowReturn GstSinkBridge::new_preroll_callback(::GstAppSink* appsink, ::gpointer user_data) {
  return static_cast<GstSinkBridge*>(user_data)->handle_buffer_common(
      appsink, gst_app_sink_pull_preroll, "preroll");
}

::GstFlowReturn GstSinkBridge::new_sample_callback(::GstAppSink* appsink, ::gpointer user_data) {
  return static_cast<GstSinkBridge*>(user_data)->handle_buffer_common(
      appsink, gst_app_sink_pull_sample, "sample");
}

}  // namespace holoscan
