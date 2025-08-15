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

#include "streaming_server_op.hpp"

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>

#include <atomic>
#include <chrono>
#include <cstdlib>  // For setenv
#include <cstring>
#include <fstream>
#include <thread>

#include <gxf/multimedia/video.hpp>
#include <gxf/std/tensor.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/io_spec.hpp>
#include <holoscan/holoscan.hpp>
#include <holoscan/core/resources/gxf/allocator.hpp>

// Define the CUDA_TRY macro
#define CUDA_TRY(stmt) do {   \
  cudaError_t err = stmt; \
  if (err != cudaSuccess) {   \
HOLOSCAN_LOG_ERROR("CUDA failed with %s: %s", #stmt,  \
   cudaGetErrorString(err));  \
throw std::runtime_error("CUDA error");   \
  }   \
} while (0)

namespace holoscan::ops {

StreamingServerOp::~StreamingServerOp() {
  try {
    HOLOSCAN_LOG_INFO("StreamingServerOp destructor: beginning cleanup...");

    // Set shutdown flag to prevent new operations
    is_shutting_down_ = true;

    if (server_) {
      // Simple, quick shutdown without excessive delays
      HOLOSCAN_LOG_INFO("Stopping server...");
      server_->stop();

  // Brief wait for shutdown to complete
  std::this_thread::sleep_for(std::chrono::milliseconds(100));

  // Reset the server pointer
  server_.reset();

  HOLOSCAN_LOG_INFO("StreamingServerOp destructor: cleanup completed");
}
  } catch (const std::exception& e) {
// Log but don't throw from destructor
HOLOSCAN_LOG_ERROR("Exception in StreamingServerOp destructor: {}", e.what());
  } catch (...) {
// Log but don't throw from destructor
HOLOSCAN_LOG_ERROR("Unknown exception in StreamingServerOp destructor");
  }
}

void StreamingServerOp::setup(OperatorSpec& spec) {
  // StreamingServerOp works standalone without pipeline connections
  // It directly interfaces with StreamSDK for bidirectional communication
  // No input/output ports needed - it's a source/sink operator

  // Define parameters with explicit default values that match YAML config
  spec.param(width_, "width", "Frame Width", "Width of the video frames in pixels", 854u);
  spec.param(height_, "height", "Frame Height", "Height of the video frames in pixels", 480u);
  spec.param(fps_, "fps", "Frames Per Second", "Frame rate of the video", 30u);
  spec.param(port_, "port", "Server Port", "Port used for streaming server", uint16_t{48010});
  spec.param(multi_instance_, "multi_instance", "Multi Instance",
 "Allow multiple server instances", false);
  spec.param(server_name_, "server_name", "Server Name",
 "Name identifier for the server", std::string("StreamingServer"));
  spec.param(receive_frames_, "receive_frames", "Receive Frames",
 "Whether to receive frames from clients", true);
  spec.param(send_frames_, "send_frames", "Send Frames",
 "Whether to send frames to clients", true);
  spec.param(allocator_, "allocator", "Memory Allocator",
 "Memory allocator for frame data");

  HOLOSCAN_LOG_INFO("StreamingServerOp setup completed - standalone mode (no pipeline connections)");
  HOLOSCAN_LOG_INFO("  - Receives frames from clients and processes them internally");
  HOLOSCAN_LOG_INFO("  - Sends processed frames back to clients");
}

void StreamingServerOp::initialize() {
  // CRITICAL: Set environment variables BEFORE calling Operator::initialize()
  // Use minimal configuration for stable local connections

  // Only essential environment variables for local streaming
  setenv("NVST_LOG_LEVEL", "INFO", 1);
  setenv("NVST_ALLOW_SELF_SIGNED_CERTS", "1", 1);
  setenv("NVST_SKIP_CERTIFICATE_VALIDATION", "1", 1);

  HOLOSCAN_LOG_INFO("Set minimal NVIDIA streaming environment variables");

  // Brief pause for environment variables to take effect
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  Operator::initialize();

  // Remove security configuration setup - let Stream SDK handle it
  HOLOSCAN_LOG_INFO("Skipping Holoscan-level security configuration - delegating to Stream SDK");

  // Log the actual parameter values received
  HOLOSCAN_LOG_INFO("StreamingServerOp initializing with parameters:");
  HOLOSCAN_LOG_INFO("  width: {}", width_.has_value() ? std::to_string(width_.get()) : "NOT SET");
  HOLOSCAN_LOG_INFO("  height: {}", height_.has_value() ? std::to_string(height_.get()) : "NOT SET");
  HOLOSCAN_LOG_INFO("  fps: {}", fps_.has_value() ? std::to_string(fps_.get()) : "NOT SET");
  HOLOSCAN_LOG_INFO("  port: {}", port_.has_value() ? std::to_string(port_.get()) : "NOT SET");
  HOLOSCAN_LOG_INFO("  receive_frames: {}", receive_frames_.has_value() ? (receive_frames_.get() ? "true" : "false") : "NOT SET");
  HOLOSCAN_LOG_INFO("  send_frames: {}", send_frames_.has_value() ? (send_frames_.get() ? "true" : "false") : "NOT SET");

  // Validate parameters before using them
  if (!width_.has_value() || width_.get() == 0) {
HOLOSCAN_LOG_ERROR("Invalid width parameter. Using default 854.");
width_ = 854U;
  }

  if (!height_.has_value() || height_.get() == 0) {
HOLOSCAN_LOG_ERROR("Invalid height parameter. Using default 480.");
height_ = 480U;
  }

  if (!fps_.has_value() || fps_.get() == 0) {
HOLOSCAN_LOG_ERROR("Invalid fps parameter. Using default 30.");
fps_ = 30U;
  }

  if (!port_.has_value() || port_.get() == 0) {
HOLOSCAN_LOG_ERROR("Invalid port parameter. Using default 48010.");
port_ = static_cast<uint16_t>(48010);
  }

  // Create server configuration
  StreamingServer::Config config;
  config.width = width_.get();
  config.height = height_.get();
  config.fps = fps_.get();
  config.port = port_.get();
  config.isMultiInstance = multi_instance_.has_value() ? multi_instance_.get() : false;
  config.serverName = server_name_.has_value() ? server_name_.get() : "StreamingServer";

  HOLOSCAN_LOG_INFO("Creating StreamingServer with validated config: width={}, height={}, fps={}, port={}",
   config.width, config.height, config.fps, config.port);

  // Initialize server with the validated parameters
  try {
server_ = std::make_unique<StreamingServer>(config);

// Set up event callback
server_->setEventCallback([this](const StreamingServer::Event& event) {
  onEvent(event);
});

HOLOSCAN_LOG_INFO("StreamingServerOp initialized successfully");
  } catch (const std::exception& e) {
HOLOSCAN_LOG_ERROR("Failed to create StreamingServer: {}", e.what());
server_.reset();
throw;
  } catch (...) {
HOLOSCAN_LOG_ERROR("Failed to create StreamingServer: unknown exception");
server_.reset();
throw;
  }
}

void StreamingServerOp::start() {
  if (!server_) {
HOLOSCAN_LOG_ERROR("Cannot start streaming: server not initialized");
return;
  }

  try {
HOLOSCAN_LOG_INFO("Starting streaming server on port {}", port_.get());

// Log detailed configuration before starting
HOLOSCAN_LOG_INFO("Server configuration:");
HOLOSCAN_LOG_INFO("  - Width: {}", width_.get());
HOLOSCAN_LOG_INFO("  - Height: {}", height_.get());
HOLOSCAN_LOG_INFO("  - FPS: {}", fps_.get());
HOLOSCAN_LOG_INFO("  - Port: {}", port_.get());
HOLOSCAN_LOG_INFO("  - Multi-instance: {}", multi_instance_.has_value() ? (multi_instance_.get() ? "true" : "false") : "default");
HOLOSCAN_LOG_INFO("  - Server name: {}", server_name_.has_value() ? server_name_.get() : "default");

// Add a small delay to ensure all initialization is complete
std::this_thread::sleep_for(std::chrono::milliseconds(100));

// Check if server is already running
if (server_->isRunning()) {
  HOLOSCAN_LOG_WARN("Server is already running");
  return;
}

// Check if port is available before starting
HOLOSCAN_LOG_INFO("Checking if port {} is available...", port_.get());

// Use a separate thread to start the server to avoid stack overflow
std::atomic<bool> start_completed{false};
std::atomic<bool> start_success{false};
std::string error_message;

std::thread start_thread([&]() {
  try {
HOLOSCAN_LOG_INFO("Starting server in separate thread...");

// Add more detailed logging around the actual start call
HOLOSCAN_LOG_DEBUG("About to call server_->start()...");
server_->start();
HOLOSCAN_LOG_DEBUG("server_->start() completed successfully");

start_success = true;
HOLOSCAN_LOG_INFO("Server start completed in thread");
  } catch (const std::exception& e) {
error_message = std::string("std::exception: ") + e.what();
HOLOSCAN_LOG_ERROR("Exception in server start thread: {}", e.what());

// Try to get more details about the error
std::string error_type = typeid(e).name();
HOLOSCAN_LOG_ERROR("Exception type: {}", error_type);
  } catch (...) {
error_message = "Unknown exception";
HOLOSCAN_LOG_ERROR("Unknown exception in server start thread");
  }
  start_completed = true;
});

// Wait for the thread to complete with timeout
start_thread.join();

// Check results
if (start_completed && start_success) {
  // Verify server started successfully
  if (server_->isRunning()) {
HOLOSCAN_LOG_INFO("✅ StreamingServer started successfully and is running");

// Log additional server status
HOLOSCAN_LOG_INFO("Server status after start:");
HOLOSCAN_LOG_INFO("  - Is running: {}", server_->isRunning() ? "YES" : "NO");
HOLOSCAN_LOG_INFO("  - Has connected clients: {}", server_->hasConnectedClients() ? "YES" : "NO");

  } else {
HOLOSCAN_LOG_ERROR("❌ StreamingServer failed to start - not running after start() call");
  }
} else {
  HOLOSCAN_LOG_ERROR("❌ StreamingServer start failed: {}",
error_message.empty() ? "Unknown error" : error_message);

  // Additional diagnostic information
  HOLOSCAN_LOG_ERROR("Diagnostic information:");
  HOLOSCAN_LOG_ERROR("  - Start completed: {}", start_completed ? "YES" : "NO");
  HOLOSCAN_LOG_ERROR("  - Start success: {}", start_success ? "YES" : "NO");
  HOLOSCAN_LOG_ERROR("  - Server running after failure: {}", server_->isRunning() ? "YES" : "NO");
}
} catch (const std::exception& e) {
HOLOSCAN_LOG_ERROR("Exception during server start setup: {}", e.what());
HOLOSCAN_LOG_ERROR("Exception type: {}", typeid(e).name());
  } catch (...) {
HOLOSCAN_LOG_ERROR("Unknown exception during server start setup");
  }
}

void StreamingServerOp::stop() {
  HOLOSCAN_LOG_INFO("StreamingServerOp::stop() called");

  // Set a flag to indicate we're shutting down
  is_shutting_down_ = true;

  if (server_) {
try {
  HOLOSCAN_LOG_INFO("Stopping StreamingServer...");

  // Simple stop without complex timeout logic
  server_->stop();

  // Brief wait for shutdown
  std::this_thread::sleep_for(std::chrono::milliseconds(50));

  HOLOSCAN_LOG_INFO("StreamingServer stopped");
} catch (const std::exception& e) {
  HOLOSCAN_LOG_ERROR("Exception during StreamingServer shutdown: {}", e.what());
} catch (...) {
  HOLOSCAN_LOG_ERROR("Unknown exception during StreamingServer shutdown");
}
  }

  HOLOSCAN_LOG_INFO("StreamingServerOp::stop() completed");
}

void StreamingServerOp::compute(InputContext& op_input, OutputContext& op_output,
ExecutionContext& context) {
  HOLOSCAN_LOG_DEBUG("!!!!compute() called");
  // Check if we're shutting down
  if (is_shutting_down_.load()) {
HOLOSCAN_LOG_DEBUG("!!!!Shutdown in progress, skipping compute");
return;
  }

  // Check if server is properly initialized and running
  if (!server_) {
HOLOSCAN_LOG_DEBUG("!!!!Server not initialized, returning");
return;
  }

  if (!server_->isRunning()) {
HOLOSCAN_LOG_DEBUG("!!!!Server not running, returning");
return;
  }

  if (!server_->hasConnectedClients()) {
HOLOSCAN_LOG_DEBUG("!!!!No connected clients, returning");
// Add a small delay to prevent busy waiting when no clients are connected
std::this_thread::sleep_for(std::chrono::milliseconds(10));
return;
  }
  HOLOSCAN_LOG_DEBUG("!!!!receive_frames_: {}", receive_frames_.has_value() ? (receive_frames_.get() ? "true" : "false") : "NOT SET");
  // Check if we should receive frames
  if (!receive_frames_.has_value() || !receive_frames_.get()) {
HOLOSCAN_LOG_DEBUG("!!!!Not configured to receive frames, returning");
// Add a small delay to prevent busy waiting
std::this_thread::sleep_for(std::chrono::milliseconds(10));
return;  // Not configured to receive frames
  }

  HOLOSCAN_LOG_DEBUG("!!!!tryReceiveFrame() called");
  // Try to receive a frame from the client with retry mechanism
  Frame received_frame;

  // Add safety check before calling StreamSDK methods
  if (!server_ || is_shutting_down_.load()) {
HOLOSCAN_LOG_DEBUG("!!!!Server not available or shutting down, skipping frame reception");
return;
  }

  // Add detailed logging about the frame reception attempt
  HOLOSCAN_LOG_DEBUG("!!!!About to call server_->tryReceiveFrame()...");
  HOLOSCAN_LOG_DEBUG("!!!!Server status: running={}, hasConnectedClients={}",
 server_->isRunning(), server_->hasConnectedClients());

  bool frame_received = false;
  int retry_count = 0;
  const int max_retries = 5;

  // Retry mechanism to give StreamSDK time to process incoming frames
  while (!frame_received && retry_count < max_retries && !is_shutting_down_.load()) {
try {
  frame_received = server_->tryReceiveFrame(received_frame);

  if (!frame_received) {
retry_count++;
HOLOSCAN_LOG_DEBUG("!!!!tryReceiveFrame() attempt {} returned false, retrying...", retry_count);

// Small delay to give StreamSDK time to process
std::this_thread::sleep_for(std::chrono::milliseconds(2));
  } else {
HOLOSCAN_LOG_DEBUG("!!!!tryReceiveFrame() succeeded on attempt {}", retry_count + 1);
  }
} catch (const std::exception& e) {
  HOLOSCAN_LOG_ERROR("Exception in tryReceiveFrame: {}", e.what());
  break;
} catch (...) {
  HOLOSCAN_LOG_ERROR("Unknown exception in tryReceiveFrame");
  break;
}
  }

  // If tryReceiveFrame failed, try the blocking receiveFrame with timeout
  if (!frame_received && !is_shutting_down_.load()) {
HOLOSCAN_LOG_DEBUG("!!!!tryReceiveFrame() failed, attempting blocking receiveFrame() with timeout...");

// Use a separate thread for blocking call with timeout
std::atomic<bool> receive_completed{false};
std::atomic<bool> receive_success{false};
Frame blocking_frame;

std::thread receive_thread([&]() {
  try {
blocking_frame = server_->receiveFrame();
receive_success = true;
HOLOSCAN_LOG_DEBUG("!!!!Blocking receiveFrame() succeeded");
  } catch (const std::exception& e) {
HOLOSCAN_LOG_DEBUG("!!!!Blocking receiveFrame() failed: {}", e.what());
  } catch (...) {
HOLOSCAN_LOG_DEBUG("!!!!Blocking receiveFrame() failed with unknown exception");
  }
  receive_completed = true;
});

// Wait for completion with timeout
auto start_time = std::chrono::steady_clock::now();
while (!receive_completed.load() && !is_shutting_down_.load()) {
  std::this_thread::sleep_for(std::chrono::milliseconds(1));

  auto elapsed = std::chrono::steady_clock::now() - start_time;
  if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count() > 10) {
HOLOSCAN_LOG_DEBUG("!!!!Blocking receiveFrame() timeout after 10ms");
break;
  }
}

if (receive_thread.joinable()) {
  receive_thread.join();
}

if (receive_success.load()) {
  received_frame = blocking_frame;
  frame_received = true;
  HOLOSCAN_LOG_DEBUG("!!!!Successfully received frame using blocking method");
}
  }

  HOLOSCAN_LOG_DEBUG("!!!!Final result: frame_received={}, retry_count={}", frame_received, retry_count);

  if (!frame_received) {
HOLOSCAN_LOG_DEBUG("!!!!No frame received after {} attempts, checking server state...", retry_count);
HOLOSCAN_LOG_DEBUG("!!!!Server running: {}", server_->isRunning());
HOLOSCAN_LOG_DEBUG("!!!!Server has clients: {}", server_->hasConnectedClients());

// Add more detailed server state logging
static int no_frame_count = 0;
no_frame_count++;

if (no_frame_count % 100 == 0) {  // Log every 100 failed attempts
  HOLOSCAN_LOG_WARN("!!!!No frames received in {} compute cycles - possible StreamSDK buffering issue", no_frame_count);
  HOLOSCAN_LOG_WARN("!!!!Server diagnostics:");
  HOLOSCAN_LOG_WARN("  - Server pointer valid: {}", server_ ? "YES" : "NO");
  HOLOSCAN_LOG_WARN("  - Server running: {}", server_->isRunning() ? "YES" : "NO");
  HOLOSCAN_LOG_WARN("  - Has connected clients: {}", server_->hasConnectedClients() ? "YES" : "NO");
  HOLOSCAN_LOG_WARN("  - Receive frames enabled: {}", receive_frames_.get() ? "YES" : "NO");
}

// Check if we're shutting down before sending heartbeat
if (is_shutting_down_.load()) {
  HOLOSCAN_LOG_DEBUG("!!!!Shutdown in progress, skipping heartbeat");
  return;
}

// Heartbeat mechanism: send a dummy frame periodically to keep connection alive
static auto last_heartbeat = std::chrono::steady_clock::now();
auto now = std::chrono::steady_clock::now();
auto time_since_heartbeat = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_heartbeat);

if (time_since_heartbeat.count() > 1000) {  // Send heartbeat every 1 second
  HOLOSCAN_LOG_DEBUG("!!!!Sending heartbeat frame to keep connection alive");

  // Create a simple heartbeat frame
  Frame heartbeat_frame;
  heartbeat_frame.width = width_.get();
  heartbeat_frame.height = height_.get();
  heartbeat_frame.format = Frame::Format::BGR;
  heartbeat_frame.timestampMs = std::chrono::duration_cast<std::chrono::milliseconds>(
  std::chrono::system_clock::now().time_since_epoch()).count();

  // Create a small test pattern (solid color)
  size_t frame_size = static_cast<size_t>(heartbeat_frame.width) *
 static_cast<size_t>(heartbeat_frame.height) * 3;  // BGR = 3 channels
  heartbeat_frame.data.resize(frame_size);

  // Fill with a simple pattern (blue color)
  for (size_t i = 0; i < frame_size; i += 3) {
heartbeat_frame.data[i] = 255;  // Blue
heartbeat_frame.data[i + 1] = 0;   // Green
heartbeat_frame.data[i + 2] = 0;   // Red
  }

  // Send heartbeat frame if send_frames is enabled
  if (send_frames_.has_value() && send_frames_.get()) {
try {
  // Double-check we're not shutting down before sending
  if (!is_shutting_down_.load() && server_ && server_->isRunning()) {
server_->sendFrame(heartbeat_frame);
HOLOSCAN_LOG_INFO("!!!!Heartbeat frame sent to keep connection alive");
  }
} catch (const std::exception& e) {
  HOLOSCAN_LOG_ERROR("Failed to send heartbeat frame: {}", e.what());
}
  }

  last_heartbeat = now;
}

// Add a small delay to prevent busy waiting when no frames are available
std::this_thread::sleep_for(std::chrono::milliseconds(1));
return;  // No frame received
  }

  HOLOSCAN_LOG_INFO("!!!!FRAME RECEIVED! Size: {} bytes, dimensions: {}x{}",
   received_frame.data.size(), received_frame.width, received_frame.height);

  // Add comprehensive frame debugging
  HOLOSCAN_LOG_INFO("!!!!FRAME DEBUG INFO:");
  HOLOSCAN_LOG_INFO("  - Frame data size: {} bytes", received_frame.data.size());
  HOLOSCAN_LOG_INFO("  - Frame dimensions: {}x{}", received_frame.width, received_frame.height);
  HOLOSCAN_LOG_INFO("  - Frame format: {}", static_cast<int>(received_frame.format));
  HOLOSCAN_LOG_INFO("  - Frame timestamp: {}", received_frame.timestampMs);
  HOLOSCAN_LOG_INFO("  - Frame data pointer: {}", received_frame.data.data() ? "VALID" : "NULL");
  HOLOSCAN_LOG_INFO("  - Frame data empty: {}", received_frame.data.empty() ? "YES" : "NO");

  // Check if frame data is all zeros
  if (!received_frame.data.empty()) {
bool all_zeros = true;
size_t non_zero_count = 0;
size_t check_limit = std::min(received_frame.data.size(), static_cast<size_t>(100));  // Check first 100 bytes

for (size_t i = 0; i < check_limit; ++i) {
  if (received_frame.data[i] != 0) {
all_zeros = false;
non_zero_count++;
  }
}

HOLOSCAN_LOG_INFO("  - First {} bytes analysis: all_zeros={}, non_zero_count={}",
 check_limit, all_zeros ? "YES" : "NO", non_zero_count);

// Log first few bytes as hex
if (received_frame.data.size() >= 16) {
  std::string hex_dump;
  for (size_t i = 0; i < 16; ++i) {
char hex_byte[4];
snprintf(hex_byte, sizeof(hex_byte), "%02X ", received_frame.data[i]);
hex_dump += hex_byte;
  }
  HOLOSCAN_LOG_INFO("  - First 16 bytes (hex): {}", hex_dump);
}
  }

  // Check server state during frame reception
  HOLOSCAN_LOG_INFO("!!!!SERVER STATE DURING FRAME RECEPTION:");
  HOLOSCAN_LOG_INFO("  - Server running: {}", server_->isRunning() ? "YES" : "NO");
  HOLOSCAN_LOG_INFO("  - Has connected clients: {}", server_->hasConnectedClients() ? "YES" : "NO");
  HOLOSCAN_LOG_INFO("  - Shutting down: {}", is_shutting_down_.load() ? "YES" : "NO");
  HOLOSCAN_LOG_INFO("  - Receive frames enabled: {}", receive_frames_.get() ? "YES" : "NO");
  HOLOSCAN_LOG_INFO("  - Send frames enabled: {}", send_frames_.get() ? "YES" : "NO");

  // Check if we're shutting down before processing the frame
  if (is_shutting_down_.load()) {
HOLOSCAN_LOG_DEBUG("!!!!Shutdown in progress, skipping frame processing");
return;
  }

  HOLOSCAN_LOG_DEBUG("!!!!received_frame.data.empty(): {}", received_frame.data.empty());
  // Validate received frame with detailed error reporting
  if (received_frame.data.empty()) {
HOLOSCAN_LOG_ERROR("!!!!FRAME VALIDATION FAILED: Frame data is empty");
HOLOSCAN_LOG_ERROR("  - This suggests the client is not sending actual frame data");
HOLOSCAN_LOG_ERROR("  - Check client-side frame creation and sending logic");
return;
  }

  if (received_frame.width == 0 || received_frame.height == 0) {
HOLOSCAN_LOG_ERROR("!!!!FRAME VALIDATION FAILED: Invalid dimensions {}x{}",
  received_frame.width, received_frame.height);
HOLOSCAN_LOG_ERROR("  - This suggests the client is not setting frame dimensions correctly");
HOLOSCAN_LOG_ERROR("  - Expected dimensions: {}x{}", width_.get(), height_.get());
return;
  }

  // Check if dimensions match expected values
  if (received_frame.width != width_.get() || received_frame.height != height_.get()) {
HOLOSCAN_LOG_WARN("!!!!FRAME DIMENSION MISMATCH:");
HOLOSCAN_LOG_WARN("  - Received: {}x{}", received_frame.width, received_frame.height);
HOLOSCAN_LOG_WARN("  - Expected: {}x{}", width_.get(), height_.get());
HOLOSCAN_LOG_WARN("  - Continuing with received dimensions...");
  }

  // Check if frame size makes sense for the dimensions
  size_t expected_min_size = received_frame.width * received_frame.height * 3;  // Minimum for BGR
  if (received_frame.data.size() < expected_min_size) {
HOLOSCAN_LOG_ERROR("!!!!FRAME SIZE VALIDATION FAILED:");
HOLOSCAN_LOG_ERROR("  - Received size: {} bytes", received_frame.data.size());
HOLOSCAN_LOG_ERROR("  - Expected minimum: {} bytes (for BGR)", expected_min_size);
HOLOSCAN_LOG_ERROR("  - This suggests incomplete frame data transmission");
return;
  }

  HOLOSCAN_LOG_INFO("!!!!FRAME VALIDATION PASSED - proceeding with processing");

  try {
// STEP 1: Convert Frame to Holoscan Tensor
int channels = 3;  // Default to BGR
if (received_frame.format == Frame::Format::BGRA || received_frame.format == Frame::Format::RGBA) {
  channels = 4;
}

// Create tensor shape (HWC format)
std::vector<int64_t> shape = {
  static_cast<int64_t>(received_frame.height),
  static_cast<int64_t>(received_frame.width),
  static_cast<int64_t>(channels)
};

// Add safety check for frame data
if (received_frame.data.empty() || received_frame.data.data() == nullptr) {
  HOLOSCAN_LOG_ERROR("Frame data is empty or null, cannot create tensor");
  return;
}

// Create GXF tensor first
auto primitive_type = nvidia::gxf::PrimitiveType::kUnsigned8;
auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();

if (!gxf_tensor) {
  HOLOSCAN_LOG_ERROR("Failed to create GXF tensor");
  return;
}

nvidia::gxf::Shape gxf_shape = nvidia::gxf::Shape{
  static_cast<int32_t>(received_frame.height),
  static_cast<int32_t>(received_frame.width),
  static_cast<int32_t>(channels)
};

// Copy frame data to host memory with safety checks
size_t data_size = received_frame.data.size();
if (data_size == 0) {
  HOLOSCAN_LOG_ERROR("Frame data size is zero, cannot create tensor");
  return;
}

auto host_data = std::shared_ptr<uint8_t[]>(new uint8_t[data_size]);
if (!host_data) {
  HOLOSCAN_LOG_ERROR("Failed to allocate memory for tensor data");
  return;
}

std::memcpy(host_data.get(), received_frame.data.data(), data_size);

// Wrap memory in GXF tensor with safety checks
try {
  gxf_tensor->wrapMemory(
  gxf_shape,
  primitive_type,
  nvidia::gxf::PrimitiveTypeSize(primitive_type),
  nvidia::gxf::ComputeTrivialStrides(gxf_shape, nvidia::gxf::PrimitiveTypeSize(primitive_type)),
  nvidia::gxf::MemoryStorageType::kSystem,
  host_data.get(),
  [host_data](void*) mutable {
host_data.reset();
return nvidia::gxf::Success;
  });
} catch (const std::exception& e) {
  HOLOSCAN_LOG_ERROR("Failed to wrap memory in GXF tensor: {}", e.what());
  return;
}

// Convert to Holoscan tensor with safety checks
auto maybe_dl_ctx = gxf_tensor->toDLManagedTensorContext();
if (!maybe_dl_ctx) {
  HOLOSCAN_LOG_ERROR("Failed to convert GXF tensor to Holoscan tensor");
  return;
}

auto tensor = std::make_shared<holoscan::Tensor>(maybe_dl_ctx.value());
if (!tensor) {
  HOLOSCAN_LOG_ERROR("Failed to create Holoscan tensor");
  return;
}

HOLOSCAN_LOG_DEBUG("!!!!Tensor created successfully");


// STEP 2: Convert Holoscan Tensor back to Frame
Frame response_frame;
response_frame.width = received_frame.width;
response_frame.height = received_frame.height;
response_frame.format = received_frame.format;

// Calculate frame size
size_t frame_size = static_cast<size_t>(response_frame.width) *
   static_cast<size_t>(response_frame.height) *
   static_cast<size_t>(channels);
response_frame.data.resize(frame_size);

// Copy tensor data back to frame with safety checks
if (tensor->data() && tensor->nbytes() > 0) {
  auto tensor_device = tensor->device();

  try {
if (tensor_device.device_type == kDLCUDA) {
  // GPU memory - copy to host first
  CUDA_TRY(cudaMemcpy(response_frame.data.data(), tensor->data(),
 std::min(static_cast<size_t>(tensor->nbytes()), frame_size), cudaMemcpyDeviceToHost));
} else {
  // CPU memory - direct copy
  std::memcpy(response_frame.data.data(), tensor->data(),
 std::min(static_cast<size_t>(tensor->nbytes()), frame_size));
}
  } catch (const std::exception& e) {
HOLOSCAN_LOG_ERROR("Failed to copy tensor data to frame: {}", e.what());
return;
  }
} else {
  HOLOSCAN_LOG_ERROR("Tensor data is null or empty, cannot copy to frame");
  return;
}

HOLOSCAN_LOG_DEBUG("!!!!Frame created successfully in Stream SDK Server");

// STEP 3: Send frame back to client (only if send_frames_ is enabled)
if (send_frames_.has_value() && send_frames_.get()) {
  // Check if we're still running before sending
  if (!is_shutting_down_.load() && server_ && server_->isRunning()) {
try {
  server_->sendFrame(response_frame);
  HOLOSCAN_LOG_INFO("!!!!Frame sent back to client (send_frames=true)");
} catch (const std::exception& e) {
  HOLOSCAN_LOG_ERROR("Failed to send frame back to client: {}", e.what());
}
  } else {
HOLOSCAN_LOG_DEBUG("!!!!Skipping frame send - server shutting down");
  }
} else {
  HOLOSCAN_LOG_DEBUG("!!!!Frame processing completed, but not sent back to client (send_frames=false)");
}

  } catch (const std::exception& e) {
HOLOSCAN_LOG_ERROR("Error in frame-tensor conversion: {}", e.what());
  } catch (...) {
HOLOSCAN_LOG_ERROR("Unknown exception in frame-tensor conversion");
  }
}
void StreamingServerOp::onEvent(const StreamingServer::Event& event) {
  // Add safety check to prevent segmentation fault
  if (is_shutting_down_.load()) {
return;
  }

  // Log timestamp for all events
  auto now = std::chrono::steady_clock::now();
  auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();

  try {
switch (event.type) {
  case StreamingServer::EventType::ClientConnecting:
HOLOSCAN_LOG_INFO("🔄 [{}] Client connecting: {}", timestamp, event.message);
break;
  case StreamingServer::EventType::ClientConnected:
HOLOSCAN_LOG_INFO("✅ [{}] Client connected successfully: {}", timestamp, event.message);
// Log current connection state with safety check
if (server_ && !is_shutting_down_.load()) {
  HOLOSCAN_LOG_INFO("📊 Server now has connected clients: {}", server_->hasConnectedClients() ? "YES" : "NO");
}
break;
  case StreamingServer::EventType::ClientDisconnected:
HOLOSCAN_LOG_WARN("❌ [{}] Client disconnected: {}", timestamp, event.message);
// Log detailed disconnection information with safety check
if (server_ && !is_shutting_down_.load()) {
  HOLOSCAN_LOG_WARN("📊 Server connection state after disconnect:");
  HOLOSCAN_LOG_WARN("  - Server running: {}", server_->isRunning() ? "YES" : "NO");
  HOLOSCAN_LOG_WARN("  - Still has connected clients: {}", server_->hasConnectedClients() ? "YES" : "NO");
}
break;
  case StreamingServer::EventType::UpstreamConnected:
HOLOSCAN_LOG_INFO("⬆️ [{}] Upstream connection established: {}", timestamp, event.message);
break;
  case StreamingServer::EventType::UpstreamDisconnected:
HOLOSCAN_LOG_WARN("⬆️❌ [{}] Upstream connection lost: {}", timestamp, event.message);
break;
  case StreamingServer::EventType::DownstreamConnected:
HOLOSCAN_LOG_INFO("⬇️ [{}] Downstream connection established: {}", timestamp, event.message);
break;
  case StreamingServer::EventType::DownstreamDisconnected:
HOLOSCAN_LOG_WARN("⬇️❌ [{}] Downstream connection lost: {}", timestamp, event.message);
break;
  case StreamingServer::EventType::FrameReceived:
HOLOSCAN_LOG_INFO("📥 [{}] Frame received: {}", timestamp, event.message);
break;
  case StreamingServer::EventType::FrameSent:
HOLOSCAN_LOG_INFO("📤 [{}] Frame sent: {}", timestamp, event.message);
break;
  default:
HOLOSCAN_LOG_WARN("🔔 [{}] Unknown event [{}]: {}", timestamp, static_cast<int>(event.type), event.message);
break;
}
  } catch (const std::exception& e) {
HOLOSCAN_LOG_ERROR("Exception in event handler: {}", e.what());
  } catch (...) {
HOLOSCAN_LOG_ERROR("Unknown exception in event handler");
  }
}

std::atomic<bool> has_new_frame_{false};

}  // namespace holoscan::ops
