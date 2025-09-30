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

#include "streaming_server_downstream_op.hpp"

#include <cstdlib>  // For setenv
#include <cstring>
#include <thread>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "frame_debug_utils.hpp"

namespace holoscan::ops {


StreamingServerDownstreamOp::~StreamingServerDownstreamOp() {
  try {
    HOLOSCAN_LOG_INFO("StreamingServerDownstreamOp destructor: beginning cleanup...");

    // Set shutdown flag to prevent new operations
    is_shutting_down_ = true;

    HOLOSCAN_LOG_INFO("StreamingServerDownstreamOp destructor: cleanup completed");
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception in StreamingServerDownstreamOp destructor: {}", e.what());
  } catch (...) {
    HOLOSCAN_LOG_ERROR("Unknown exception in StreamingServerDownstreamOp destructor");
  }
}

void StreamingServerDownstreamOp::setup(OperatorSpec& spec) {
  // This operator only takes input holoscan::Tensor - no outputs
  spec.input<holoscan::Tensor>("input_frames");

  // Define parameters with explicit default values (can override resource defaults)
  spec.param(width_, "width", "Frame Width", "Width of the video frames in pixels", 854u);
  spec.param(height_, "height", "Frame Height", "Height of the video frames in pixels", 480u);
  spec.param(fps_, "fps", "Frames Per Second", "Frame rate of the video", 30u);
  spec.param(enable_processing_, "enable_processing", "Enable Processing",
             "Enable frame processing (mirroring, etc.)", false);
  spec.param(processing_type_, "processing_type", "Processing Type",
             "Type of processing to apply (mirror, rotate, etc.)", std::string("none"));
  spec.param(allocator_, "allocator", "Memory Allocator",
             "Memory allocator for frame data");
  spec.param(streaming_server_resource_, "streaming_server_resource", "Streaming Server Resource",
             "StreamingServerResource for managing server connections");

  HOLOSCAN_LOG_INFO("StreamingServerDownstreamOp setup completed - sends frames to clients");
}

void StreamingServerDownstreamOp::initialize() {
  Operator::initialize();

  // Get the streaming server resource
  auto streaming_server_resource = streaming_server_resource_.get();
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

  HOLOSCAN_LOG_INFO("StreamingServerDownstreamOp initializing with parameters:");
  HOLOSCAN_LOG_INFO("  - Width: {}", width_.get());
  HOLOSCAN_LOG_INFO("  - Height: {}", height_.get());
  HOLOSCAN_LOG_INFO("  - FPS: {}", fps_.get());
  HOLOSCAN_LOG_INFO("  - Processing enabled: {}", enable_processing_.has_value() ? enable_processing_.get() : false);

  try {
    // Set up event callback on the shared resource
    streaming_server_resource->set_event_callback([this](const StreamingServerResource::Event& event) {
      on_streaming_server_event(event);
    });

    HOLOSCAN_LOG_INFO("StreamingServerDownstreamOp initialized successfully");
    start_time_ticks_ = std::chrono::steady_clock::now().time_since_epoch().count();

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to initialize StreamingServerDownstreamOp: {}", e.what());
    throw;
  }
}

void StreamingServerDownstreamOp::start() {
  auto streaming_server_resource = streaming_server_resource_.get();
  if (!streaming_server_resource) {
    HOLOSCAN_LOG_ERROR("Cannot start downstream operator: StreamingServerResource not available");
    return;
  }

  try {
    HOLOSCAN_LOG_INFO("Starting downstream streaming server...");

    // Start the shared streaming server resource
    // Note: This might already be started by another operator, which is fine
    if (!streaming_server_resource->is_running()) {
      streaming_server_resource->start();
    }

    HOLOSCAN_LOG_INFO("✅ Downstream StreamingServer started successfully");

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception during downstream server start: {}", e.what());
  }
}

void StreamingServerDownstreamOp::stop() {
  HOLOSCAN_LOG_INFO("StreamingServerDownstreamOp::stop() called");
  is_shutting_down_ = true;

  // Note: We don't stop the StreamingServerResource here as it might be shared
  // with other operators. The resource manages its own lifecycle.
}

void StreamingServerDownstreamOp::compute(InputContext& op_input, OutputContext& op_output,
                                         ExecutionContext& context) {
  if (is_shutting_down_.load()) {
    return;
  }

  auto streaming_server_resource = streaming_server_resource_.get();
  if (!streaming_server_resource || !streaming_server_resource->is_running()) {
    return;
  }

  // Receive input tensor data
  auto input_message = op_input.receive<holoscan::Tensor>("input_frames");

  // Add debugging for tensor reception
  if (!input_message) {
    HOLOSCAN_LOG_ERROR("❌ No input frames received in downstream operator!");
    return;  // No input data available
  }

  HOLOSCAN_LOG_INFO("✅ DOWNSTREAM: Received holoscan::Tensor from pipeline");

  auto& input_tensor = input_message.value();
  frames_processed_++;

  // Get tensor dimensions and size
  auto shape = input_tensor.shape();
  size_t tensor_size = input_tensor.nbytes();

  HOLOSCAN_LOG_INFO("📊 DOWNSTREAM: Processing tensor {} - shape: {}, {} bytes",
                   frames_processed_.load(), fmt::join(shape, "x"), tensor_size);

  // DEBUG: Log tensor information (every 10 frames for more frequent validation)
  static int debug_frame_counter = 0;
  debug_frame_counter++;
  if (debug_frame_counter % 10 == 0) {
    auto dtype = input_tensor.dtype();
    auto device = input_tensor.device();
    HOLOSCAN_LOG_INFO("DEBUG: Tensor info for frame {}: shape={}, dtype=({},{},{}), device=({},{})",
                     debug_frame_counter, fmt::join(shape, "x"),
                     dtype.code, dtype.bits, dtype.lanes,
                     static_cast<int>(device.device_type), device.device_id);
  }

  try {
    // Process the tensor if processing is enabled
    holoscan::Tensor processed_tensor = input_tensor;
    if (enable_processing_.has_value() && enable_processing_.get()) {
      processed_tensor = process_frame(input_tensor);
    }

    // Convert tensor to output frame format
    Frame output_frame = convert_tensor_to_frame(processed_tensor);

#ifdef HOLOSCAN_DEBUG_FRAME_WRITING
    // DEBUG: Write output Frame to disk (every 10 frames for more frequent validation)
    static int debug_output_frame_counter = 0;
    debug_output_frame_counter++;
    if (debug_output_frame_counter % 10 == 0) {
      HOLOSCAN_LOG_INFO("DEBUG: Writing output Frame to disk for frame {}", debug_output_frame_counter);
      debug_utils::writeFrameToDisk(output_frame, "debug_downstream_output", debug_output_frame_counter);
    }
#endif // HOLOSCAN_DEBUG_FRAME_WRITING

    // Send frame to connected clients via StreamingServerResource
    if (output_frame.getDataSize() > 0) {
      HOLOSCAN_LOG_INFO("📤 DOWNSTREAM: Attempting to send frame to client: {}x{}, {} bytes",
                       output_frame.getWidth(), output_frame.getHeight(), output_frame.getDataSize());

      streaming_server_resource->send_frame(output_frame);
      frames_sent_++;

      HOLOSCAN_LOG_INFO("✅ DOWNSTREAM: Frame sent successfully to StreamingServerResource");
    } else {
      HOLOSCAN_LOG_ERROR("❌ DOWNSTREAM: Cannot send frame - invalid data size: {}", output_frame.getDataSize());
    }

    // Log performance every 30 frames
    if (frames_processed_ % 30 == 0) {
      auto now = std::chrono::steady_clock::now();
      auto start_time = std::chrono::steady_clock::time_point(std::chrono::steady_clock::duration(start_time_ticks_.load()));
      auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time);
      if (elapsed.count() > 0) {
        float fps = static_cast<float>(frames_processed_) / elapsed.count();
        HOLOSCAN_LOG_INFO("📊 Downstream Performance: Processed {} frames, Sent {} frames ({:.2f} FPS)",
                         frames_processed_.load(), frames_sent_.load(), fps);
      }
    }

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error processing frame in downstream operator: {}", e.what());
  }
}

void StreamingServerDownstreamOp::on_streaming_server_event(const StreamingServerResource::Event& event) {
  if (is_shutting_down_.load()) {
    return;
  }

  auto now = std::chrono::steady_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  try {
    switch (event.type) {
      case StreamingServerResource::EventType::ClientConnected:
        HOLOSCAN_LOG_INFO("✅ [DOWNSTREAM {}] Client connected: {}", timestamp, event.message);
        break;
      case StreamingServerResource::EventType::ClientDisconnected:
        HOLOSCAN_LOG_WARN("❌ [DOWNSTREAM {}] Client disconnected: {}", timestamp, event.message);
        break;
      case StreamingServerResource::EventType::DownstreamConnected:
        HOLOSCAN_LOG_INFO("⬇️ [DOWNSTREAM {}] Downstream connection established: {}", timestamp, event.message);
        downstream_connected_ = true;
        break;
      case StreamingServerResource::EventType::DownstreamDisconnected:
        HOLOSCAN_LOG_WARN("⬇️❌ [DOWNSTREAM {}] Downstream connection lost: {}", timestamp, event.message);
        downstream_connected_ = false;
        break;
      case StreamingServerResource::EventType::FrameSent:
        HOLOSCAN_LOG_DEBUG("📤 [DOWNSTREAM {}] Frame sent: {}", timestamp, event.message);
        break;
      default:
        HOLOSCAN_LOG_DEBUG("🔔 [DOWNSTREAM {}] Event [{}]: {}", timestamp, static_cast<int>(event.type), event.message);
        break;
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Exception in downstream event handler: {}", e.what());
  }
}

holoscan::Tensor StreamingServerDownstreamOp::process_frame(const holoscan::Tensor& input_tensor) {
  if (!enable_processing_.has_value() || !enable_processing_.get()) {
    return input_tensor;  // No processing
  }

  std::string processing_type = processing_type_.has_value() ? processing_type_.get() : "none";

  if (processing_type == "mirror") {
    return mirror_horizontally(input_tensor);
  } else if (processing_type == "rotate") {
    // TODO: Implement rotation if needed
    HOLOSCAN_LOG_WARN("Rotation processing not implemented yet");
    return input_tensor;
  } else {
    return input_tensor;  // No processing
  }
}

Frame StreamingServerDownstreamOp::convert_tensor_to_frame(const holoscan::Tensor& tensor) {
  // Get tensor properties
  auto shape = tensor.shape();
  auto dtype = tensor.dtype();
  const void* data_ptr = tensor.data();
  size_t data_size = tensor.nbytes();

  // Validate tensor dimensions and layout
  // This operator expects exactly 3D tensors in HWC format: height, width, channels
  if (shape.size() != 3) {
    throw std::runtime_error(fmt::format("Expected 3D tensor in HWC format (height, width, channels), but got {}D tensor with shape [{}]",
                                        shape.size(), fmt::join(shape, ", ")));
  }

  uint32_t height = static_cast<uint32_t>(shape[0]);   // Height (first dimension)
  uint32_t width = static_cast<uint32_t>(shape[1]);    // Width (second dimension)
  uint32_t channels = static_cast<uint32_t>(shape[2]);  // Channels (third dimension)

  // Validate reasonable dimensions for video frames
  if (height == 0 || width == 0 || channels == 0) {
    throw std::runtime_error(fmt::format("Invalid tensor dimensions: height={}, width={}, channels={}", height, width, channels));
  }
  if (channels > 4) {
    throw std::runtime_error(fmt::format("Unsupported number of channels: {}. Expected 1-4 channels for video frames.", channels));
  }

  // Create VideoFrame with appropriate dimensions
  Frame output_frame(width, height);

  // Set timestamp (use current time as tensor doesn't have timestamp)
  auto now = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count();
  output_frame.setTimestamp(static_cast<uint64_t>(now));

  // Set pixel format based on number of channels
  if (channels == 3) {
    output_frame.setFormat(::PixelFormat::BGR);  // Assume BGR for 3 channels
  } else if (channels == 4) {
    output_frame.setFormat(::PixelFormat::BGRA);  // Assume BGRA for 4 channels
  } else {
    output_frame.setFormat(::PixelFormat::BGRA);  // Default fallback
  }

  // Copy tensor data to frame - only support uint8 data
  if (dtype.code == kDLUInt && dtype.bits == 8 && dtype.lanes == 1) { // uint8
    output_frame.setData(static_cast<const uint8_t*>(data_ptr), data_size);
  } else {
    throw std::runtime_error(fmt::format("Unsupported tensor data type: code={}, bits={}, lanes={}. Only uint8 tensors are supported.", dtype.code, dtype.bits, dtype.lanes));
  }

  return output_frame;
}

holoscan::Tensor StreamingServerDownstreamOp::mirror_horizontally(const holoscan::Tensor& input_tensor) {
  // This is a simplified implementation
  // In a real implementation, you would need to properly mirror the tensor data

  HOLOSCAN_LOG_DEBUG("Applying horizontal mirror processing to tensor");

  // For now, just return the input tensor
  // TODO: Implement actual horizontal mirroring using tensor operations
  return input_tensor;
}

} // namespace holoscan::ops
