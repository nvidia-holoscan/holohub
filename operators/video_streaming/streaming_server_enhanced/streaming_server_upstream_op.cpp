/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "streaming_server_upstream_op.hpp"

#include <algorithm>
#include <cstdlib>    // For setenv
#include <cstring>
#include <thread>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <vector>
#include <fstream>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <gxf/std/tensor.hpp>
#include <gxf/core/handle.hpp>

#include "frame_debug_utils.hpp"

namespace holoscan::ops {

StreamingServerUpstreamOp::~StreamingServerUpstreamOp() {
  try {
    HOLOSCAN_LOG_INFO("StreamingServerUpstreamOp destructor: beginning cleanup...");

      // Set shutdown flag to prevent new operations
    is_shutting_down_ = true;

    HOLOSCAN_LOG_INFO("StreamingServerUpstreamOp destructor: cleanup completed");
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception in StreamingServerUpstreamOp destructor: {}", e.what());
  } catch (...) {
    HOLOSCAN_LOG_ERROR("Unknown exception in StreamingServerUpstreamOp destructor");
  }
}

void StreamingServerUpstreamOp::setup(OperatorSpec& spec) {
    // This operator only outputs holoscan::Tensor - no inputs
  spec.output<holoscan::Tensor>("output_frames");

    // Define parameters with explicit default values (can override resource defaults)
  spec.param(width_, "width", "Frame Width", "Width of the video frames in pixels", 854u);
  spec.param(height_, "height", "Frame Height", "Height of the video frames in pixels", 480u);
  spec.param(fps_, "fps", "Frames Per Second", "Frame rate of the video", 30u);
  spec.param(allocator_, "allocator", "Memory Allocator",
             "Memory allocator for frame data");
  spec.param(streaming_server_resource_, "streaming_server_resource", "Streaming Server Resource",
             "StreamingServerResource for managing server connections");

  HOLOSCAN_LOG_INFO("StreamingServerUpstreamOp setup completed - receives frames from clients");
}

void StreamingServerUpstreamOp::initialize() {
  Operator::initialize();

    // Get the streaming server resource
  if (!streaming_server_resource) {
    throw std::runtime_error("StreamingServerResource is null");
  }

    // Validate parameters and use resource defaults if not specified
  auto resource_config = streaming_server_resource->get_config();

  if (!width_.has_value() || width_.get() == 0) {
    HOLOSCAN_LOG_INFO("Using resource width: {}", resource_config.width);
    width_ = resource_config.width;
  }

  if (!height_.has_value() || height_.get() == 0) {
    HOLOSCAN_LOG_INFO("Using resource height: {}", resource_config.height);
    height_ = resource_config.height;
  }

  if (!fps_.has_value() || fps_.get() == 0) {
    HOLOSCAN_LOG_INFO("Using resource fps: {}", resource_config.fps);
    fps_ = resource_config.fps;
  }

  HOLOSCAN_LOG_INFO("StreamingServerUpstreamOp initializing with parameters:");
  HOLOSCAN_LOG_INFO("  - Width: {}", width_.get());
  HOLOSCAN_LOG_INFO("  - FPS: {}", fps_.get());

  try {
      // Set up event callback on the shared resource
    streaming_server_resource->set_event_callback(
        [this](const StreamingServerResource::Event& event) {
      on_streaming_server_event(event);
    });

      // Note: Frame received callback is not available in the library
      // We'll use polling approach with tryReceiveFrame instead

    HOLOSCAN_LOG_INFO("StreamingServerUpstreamOp initialized successfully");
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to initialize StreamingServerUpstreamOp: {}", e.what());
    throw;
  }
}

  auto streaming_server_resource = streaming_server_resource_.get();
  if (!streaming_server_resource) {
    HOLOSCAN_LOG_ERROR("Cannot start upstream operator: StreamingServerResource not available");
    return;
  }

  try {
    HOLOSCAN_LOG_INFO("Starting upstream streaming server...");

      // Start the shared streaming server resource
      // Note: This might already be started by another operator, which is fine
    if (!streaming_server_resource->is_running()) {
      streaming_server_resource->start();
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception during upstream server start: {}", e.what());
  }
}

void StreamingServerUpstreamOp::stop() {
  HOLOSCAN_LOG_INFO("StreamingServerUpstreamOp::stop() called");
  is_shutting_down_ = true;

    // Note: We don't stop the StreamingServerResource here as it might be shared
    // with other operators. The resource manages its own lifecycle.
}

void StreamingServerUpstreamOp::compute(InputContext& op_input, OutputContext& op_output,
                                       ExecutionContext& context) {
  if (is_shutting_down_.load()) {
    return;
  }

  auto streaming_server_resource = streaming_server_resource_.get();
  if (!streaming_server_resource || !streaming_server_resource->is_running()) {
    return;
  }

    // Try to receive a frame using polling approach
  Frame received_frame;
  if (streaming_server_resource->try_receive_frame(received_frame)) {
    frames_received_++;

      // 🔍 DUPLICATE DETECTION: Check if this frame was already processed
    if (is_duplicate_frame(received_frame)) {
      HOLOSCAN_LOG_WARN("⚠️  Skipping duplicate frame with timestamp {}",
                         received_frame.getTimestamp());
      return;    // Skip processing this duplicate frame
    }

    // add loggign to check if we received the frame
    HOLOSCAN_LOG_INFO("✅ Processing UNIQUE frame: {}x{}, {} bytes, timestamp={}",
                      received_frame.getWidth(),
                      received_frame.getHeight(),
                      received_frame.getDataSize(),
                      received_frame.getTimestamp());

      // Log duplicate detection statistics every 30 frames
    static int stats_counter = 0;
    stats_counter++;
    if (stats_counter % 30 == 0) {
      HOLOSCAN_LOG_INFO("📊 Frame Processing Stats: Total={}, Unique={}, Duplicates={}",
                       frames_received_.load(),
                       unique_frames_processed_.load(),
                       duplicate_frames_detected_.load());
    }

      // Only write unique frames to disk (use unique frame counter for consistent numbering)
    uint64_t unique_count = unique_frames_processed_.load();

#ifdef HOLOSCAN_DEBUG_FRAME_WRITING
      // DEBUG: Write received Frame to disk (every 10 frames for more frequent validation)
    static int debug_frame_counter = 0;
    debug_frame_counter++;

    if (unique_count % 10 == 0) {
      HOLOSCAN_LOG_INFO("💾 DEBUG: Writing UNIQUE received Frame to disk (unique frame {})",
                         unique_count);
      debug_utils::writeFrameToDisk(received_frame, "debug_upstream_received",
                                     static_cast<int>(unique_count));
    }
#endif  // HOLOSCAN_DEBUG_FRAME_WRITING

      // Convert Frame to holoscan::Tensor
    holoscan::Tensor output_tensor = convert_frame_to_tensor(received_frame);

      // DEBUG: Log converted tensor information (every 10 unique frames)
    if (unique_count % 10 == 0) {
      auto shape = output_tensor.shape();
      auto dtype = output_tensor.dtype();
      auto device = output_tensor.device();
      HOLOSCAN_LOG_INFO("💾 DEBUG: Converted tensor info (unique frame {}): shape={}, "
                         "dtype=({},{},{}), device=({},{})",
                         unique_count, fmt::join(shape, "x"),
                       dtype.code, dtype.bits, dtype.lanes,
                       static_cast<int>(device.device_type), device.device_id);
    }

    if (output_tensor.data() != nullptr) {
        // Output the tensor
      op_output.emit(output_tensor, "output_frames");

      auto shape = output_tensor.shape();
      HOLOSCAN_LOG_DEBUG("Emitted tensor: shape={}, {} bytes",
                        fmt::join(shape, "x"), output_tensor.nbytes());

        // Log performance every 30 frames
      if (frames_received_ % 30 == 0) {
        auto now = std::chrono::steady_clock::now();
        auto start_time = std::chrono::steady_clock::time_point(
            std::chrono::steady_clock::duration(start_time_ticks_.load()));
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
        if (elapsed.count() > 0) {
          float fps = static_cast<float>(frames_received_) / elapsed.count();
          HOLOSCAN_LOG_INFO("📊 Upstream Performance: Received {} frames ({:.2f} FPS)",
                             frames_received_.load(), fps);
        }
    }
  }
}

void StreamingServerUpstreamOp::on_streaming_server_event(
    const StreamingServerResource::Event& event) {
  if (is_shutting_down_.load()) {
    return;
  }

  auto now = std::chrono::steady_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
      now.time_since_epoch()).count();

  try {
    switch (event.type) {
      case StreamingServerResource::EventType::ClientConnecting:
        HOLOSCAN_LOG_INFO("🔄 [UPSTREAM {}] Client connecting: {}",
                           timestamp, event.message);
        break;
      case StreamingServerResource::EventType::ClientConnected:
        HOLOSCAN_LOG_INFO("✅ [UPSTREAM {}] Client connected: {}",
                           timestamp, event.message);
        break;
      case StreamingServerResource::EventType::ClientDisconnected:
        HOLOSCAN_LOG_WARN("❌ [UPSTREAM {}] Client disconnected: {}",
                           timestamp, event.message);
        break;
      case StreamingServerResource::EventType::UpstreamConnected:
        HOLOSCAN_LOG_INFO("⬆️ [UPSTREAM {}] Upstream connection established: {}",
                           timestamp, event.message);
        upstream_connected_ = true;
        break;
      case StreamingServerResource::EventType::UpstreamDisconnected:
        HOLOSCAN_LOG_WARN("⬆️❌ [UPSTREAM {}] Upstream connection lost: {}",
                           timestamp, event.message);
        upstream_connected_ = false;
        break;
      case StreamingServerResource::EventType::FrameReceived:
        HOLOSCAN_LOG_DEBUG("📥 [UPSTREAM {}] Frame received: {}",
                            timestamp, event.message);
        break;
      default:
        HOLOSCAN_LOG_DEBUG("🔔 [UPSTREAM {}] Event [{}]: {}",
                            timestamp, static_cast<int>(event.type), event.message);
        break;
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception in upstream event handler: {}", e.what());
  }
}

holoscan::Tensor StreamingServerUpstreamOp::convert_frame_to_tensor(const Frame& frame) {
  if (!frame.isValid()) {
      // Return empty tensor for invalid frame
    return holoscan::Tensor();
  }

    // Get frame properties
  uint32_t width = frame.getWidth();
  uint32_t height = frame.getHeight();
  size_t data_size = frame.getDataSize();
  const uint8_t* frame_data = frame.getData();
  auto frame_format = frame.getFormat();

    // Determine number of channels based on pixel format
  uint32_t channels;
  switch (frame_format) {
    case ::PixelFormat::BGR:
      channels = 3;
      break;
    case ::PixelFormat::BGRA:
    case ::PixelFormat::RGBA:
      channels = 4;
      break;
    default:
      channels = 4;    // Default to 4 channels
      break;
  }

    // Create tensor shape [height, width, channels] (HWC format)
  std::vector<int64_t> shape = {static_cast<int64_t>(height),
                                static_cast<int64_t>(width),
                                static_cast<int64_t>(channels)};

    // Calculate expected size
  size_t expected_size = height * width * channels;
  if (data_size != expected_size) {
    HOLOSCAN_LOG_WARN("Frame data size mismatch: expected {}, got {}", expected_size, data_size);
  }

    // Create GXF tensor first (following the correct pattern)
  auto primitive_type = nvidia::gxf::PrimitiveType::kUnsigned8;
  auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();

  if (!gxf_tensor) {
    HOLOSCAN_LOG_ERROR("Failed to create GXF tensor");
    return holoscan::Tensor();
  }

  nvidia::gxf::Shape gxf_shape = nvidia::gxf::Shape{
    static_cast<int32_t>(height),
    static_cast<int32_t>(width),
    static_cast<int32_t>(channels)
  };

    // Copy frame data to host memory
  auto host_data = std::shared_ptr<uint8_t[]>(new uint8_t[data_size]);
  if (!host_data) {
    HOLOSCAN_LOG_ERROR("Failed to allocate memory for tensor data");
    return holoscan::Tensor();
  }

  std::memcpy(host_data.get(), frame_data, data_size);

    // Wrap memory in GXF tensor
  try {
    gxf_tensor->wrapMemory(
        gxf_shape,
        primitive_type,
        nvidia::gxf::PrimitiveTypeSize(primitive_type),
        nvidia::gxf::ComputeTrivialStrides(gxf_shape,
                                            nvidia::gxf::PrimitiveTypeSize(primitive_type)),
        nvidia::gxf::MemoryStorageType::kSystem,
        host_data.get(),
        [host_data](void*) mutable {
          host_data.reset();
          return nvidia::gxf::Success;
        });
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to wrap memory in GXF tensor: {}", e.what());
    return holoscan::Tensor();
  }

    // Convert to Holoscan tensor
  auto maybe_dl_ctx = gxf_tensor->toDLManagedTensorContext();
  if (!maybe_dl_ctx) {
    HOLOSCAN_LOG_ERROR("Failed to convert GXF tensor to Holoscan tensor");
    return holoscan::Tensor();
  }

  return holoscan::Tensor(maybe_dl_ctx.value());
}

bool StreamingServerUpstreamOp::is_duplicate_frame(const Frame& frame) {
  uint64_t frame_timestamp = frame.getTimestamp();

    // Quick check: if timestamp is same as last processed, it's definitely a duplicate
  if (frame_timestamp == last_processed_timestamp_.load()) {
    duplicate_frames_detected_++;
    HOLOSCAN_LOG_DEBUG("Duplicate frame detected: same timestamp as last processed ({})",
                        frame_timestamp);
    return true;
  }

    // More comprehensive check: maintain a set of recent timestamps
  std::lock_guard<std::mutex> lock(frame_tracking_mutex_);

    // Check if we've already processed this exact timestamp
  if (processed_frame_timestamps_.find(frame_timestamp) != processed_frame_timestamps_.end()) {
    duplicate_frames_detected_++;
    HOLOSCAN_LOG_WARN("🔍 DUPLICATE FRAME DETECTED: timestamp {} already processed",
                       frame_timestamp);
    return true;
  }

    // Add to processed set
  processed_frame_timestamps_.insert(frame_timestamp);

    // Keep only recent timestamps (last 1000) to prevent memory growth
  if (processed_frame_timestamps_.size() > 1000) {
      // Remove timestamps older than the most recent 900 (keep some buffer)
      // Find the 900th most recent timestamp as the cutoff
    std::vector<uint64_t> sorted_timestamps(processed_frame_timestamps_.begin(),
                                             processed_frame_timestamps_.end());
    std::sort(sorted_timestamps.begin(), sorted_timestamps.end(), std::greater<uint64_t>());

    if (sorted_timestamps.size() > 900) {
      uint64_t cutoff_timestamp = sorted_timestamps[899];    // Keep top 900 (0-indexed)

        // Remove all timestamps older than cutoff
      auto it = processed_frame_timestamps_.begin();
      while (it != processed_frame_timestamps_.end()) {
        if (*it < cutoff_timestamp) {
          it = processed_frame_timestamps_.erase(it);
        } else {
          ++it;
        }
      }

      HOLOSCAN_LOG_DEBUG("Timestamp cleanup: kept {} recent timestamps, removed older ones",
                        processed_frame_timestamps_.size());
    }
  }

    // Update last processed timestamp
  last_processed_timestamp_.store(frame_timestamp);
  unique_frames_processed_++;

  HOLOSCAN_LOG_DEBUG("✅ Unique frame accepted: timestamp {}", frame_timestamp);
  return false;
}
}  // namespace holoscan::ops
