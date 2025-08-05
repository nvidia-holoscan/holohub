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

#include "streaming_client.hpp"

#include <cuda_runtime.h>
#include <dlpack/dlpack.h>
#include <fmt/format.h>
#include <gxf/core/gxf.h>

#include <algorithm>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <gxf/std/tensor.hpp>
#include <holoscan/core/conditions/gxf/asynchronous.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/holoscan.hpp>

#include "VideoFrame.h"


// Define the CUDA_TRY macro
#define CUDA_TRY(stmt) do {                                       \
  cudaError_t err = stmt;                                         \
  if (err != cudaSuccess) {                                       \
    HOLOSCAN_LOG_ERROR("CUDA failed with %s: %s", #stmt,          \
                       cudaGetErrorString(err));                  \
    throw std::runtime_error("CUDA error");                       \
  }                                                               \
} while (0)

namespace holoscan::ops {

void StreamingClientOp::setup(holoscan::OperatorSpec& spec) {
  // Define inputs/outputs
  spec.input<holoscan::gxf::Entity>("input_frames");
  spec.output<holoscan::gxf::Entity>("output_frames").condition(ConditionType::kNone);

  // Define parameters with explicit default values that match server
  spec.param(width_, "width", "Frame Width", "Width of the video frames in pixels", 854u);
  spec.param(height_, "height", "Frame Height", "Height of the video frames in pixels", 480u);
  spec.param(fps_, "fps", "Frames Per Second", "Frame rate of the video", 30u);
  spec.param(server_ip_, "server_ip", "Server IP", "IP address of the streaming server", 
             std::string("127.0.0.1"));
  spec.param(signaling_port_, "signaling_port", "Signaling Port", 
             "Port used for signaling", uint16_t{48010});  // Match StreamSDK hardcoded port
  spec.param(receive_frames_, "receive_frames", "Receive Frames", 
             "Whether to receive frames from server", true);
  spec.param(send_frames_, "send_frames", "Send Frames", 
             "Whether to send frames to server", true);

  // Print the parameters for debugging with correct values
  HOLOSCAN_LOG_INFO("StreamingClientOp setup with defaults: width={}, height={}, fps={}, server_ip={}, port={}, send_frames={}",
                    854u, 480u, 30u, "127.0.0.1", 48010, true);
}

void StreamingClientOp::initialize() {
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

  // Validate parameters before using them
  if (width_.get() == 0 || height_.get() == 0) {
    HOLOSCAN_LOG_ERROR("Invalid dimensions: width={}, height={}. Using defaults 854x480.",
                      width_.get(), height_.get());
    width_ = 854u;
    height_ = 480u;
  }

  if (fps_.get() == 0) {
    HOLOSCAN_LOG_ERROR("Invalid fps: {}. Using default 30.", fps_.get());
    fps_ = 30u;
  }

  if (server_ip_.get().empty()) {
    HOLOSCAN_LOG_ERROR("Invalid server IP. Using default 127.0.0.1");
    server_ip_ = "127.0.0.1";
  }

  HOLOSCAN_LOG_INFO("Creating StreamingClient with: width={}, height={}, fps={}, port={}",
                   width_.get(), height_.get(), fps_.get(), signaling_port_.get());
  
  // Initialize timing control
  frame_interval_ = std::chrono::microseconds(1000000 / fps_.get()); // Convert fps to microseconds
  last_frame_time_ = std::chrono::steady_clock::now();
  
  // Initialize client with the validated parameters
  client_ = std::make_unique<StreamingClient>(
      width_.get(), height_.get(), fps_.get(), signaling_port_.get());
  
  // Set up frame callback if we're receiving frames
  if (receive_frames_.get()) {
    client_->setFrameCallback([this](const VideoFrame& frame) {
      HOLOSCAN_LOG_INFO("Received frame: {}x{}", frame.getWidth(), frame.getHeight());
      onFrameReceived(frame);
    });
  }
  
  // Set up frame source if we're sending frames
  if (send_frames_.get()) {
    client_->setFrameSource([this]() {
      return generateFrame();
    });
  }
  
  HOLOSCAN_LOG_INFO("StreamingClientOp initialized successfully");
}

void StreamingClientOp::start() {
  if (!client_) {
    HOLOSCAN_LOG_ERROR("Cannot start streaming: client not initialized");
    return;
  }

  try {
    // Add a small delay to ensure environment variables are set
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    
    // Start streaming with the configured parameters
    HOLOSCAN_LOG_INFO("Starting streaming with server: {}:{}", server_ip_.get(), signaling_port_.get());
    
    // Add connection timeout and retry logic with shorter delays
    int max_retries = 5;
    int retry_count = 0;
    bool connection_successful = false;
    
    while (retry_count < max_retries && !connection_successful) {
      try {
        HOLOSCAN_LOG_INFO("Connection attempt {} of {}", retry_count + 1, max_retries);
        client_->startStreaming(server_ip_.get(), signaling_port_.get());
        connection_successful = true;
        HOLOSCAN_LOG_INFO("StreamingClient started streaming successfully");
        
        // Give the connection time to stabilize
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        
      } catch (const std::exception& e) {
        retry_count++;
        HOLOSCAN_LOG_WARN("Connection attempt {} failed: {}. Retrying in 500ms...", retry_count, e.what());
        if (retry_count >= max_retries) {
          HOLOSCAN_LOG_ERROR("All connection attempts failed. Final error: {}", e.what());
          throw;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
      }
    }
    
    // Wait for first frame if we're receiving
    if (receive_frames_.get() && connection_successful) {
      HOLOSCAN_LOG_INFO("Waiting for first frame from server...");
      if (client_->waitForFirstFrameReceived()) {
        HOLOSCAN_LOG_INFO("Received first frame from server");
      } else {
        HOLOSCAN_LOG_WARN("Timeout waiting for first frame from server");
      }
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to start streaming client: {}", e.what());
    throw; // Re-throw to indicate failure
  } catch (...) {
    HOLOSCAN_LOG_ERROR("Unknown exception during client start");
    throw;
  }
}

void StreamingClientOp::stop() {
  if (client_) {
    try {
      HOLOSCAN_LOG_INFO("Stopping StreamingClient...");
      
      // Clear frame callback to prevent race conditions
      if (receive_frames_.get()) {
        client_->setFrameCallback(nullptr);
      }
      
      // Clear frame source to prevent race conditions
      if (send_frames_.get()) {
        client_->setFrameSource(nullptr);
      }
      
      // Stop streaming with timeout
      client_->stopStreaming();
      
      // Wait for streaming to end with timeout
      auto stop_timeout = std::chrono::steady_clock::now() + std::chrono::seconds(5);
      bool stopped = false;
      
      while (!stopped && std::chrono::steady_clock::now() < stop_timeout) {
        try {
          stopped = client_->waitForStreamingEnded();
          if (!stopped) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
        } catch (const std::exception& e) {
          HOLOSCAN_LOG_WARN("Exception while waiting for streaming to end: {}", e.what());
          break;
        }
      }
      
      if (!stopped) {
        HOLOSCAN_LOG_WARN("StreamingClient did not stop within timeout, forcing shutdown");
      }
      
      HOLOSCAN_LOG_INFO("StreamingClient stopped streaming");
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Error stopping StreamingClient: {}", e.what());
    } catch (...) {
      HOLOSCAN_LOG_ERROR("Unknown error stopping StreamingClient");
    }
    
    // Reset client pointer to prevent double cleanup
    client_.reset();
  }
}

void StreamingClientOp::compute(holoscan::InputContext& op_input,
                             holoscan::OutputContext& op_output,
                             holoscan::ExecutionContext& context) {
    HOLOSCAN_LOG_INFO("!!!!COMPUTE METHOD CALLED - checking for input tensors");
    
    // Check if we have any input messages
    auto input_message = op_input.receive<holoscan::gxf::Entity>("input_frames");
    if (!input_message) {
        HOLOSCAN_LOG_ERROR("!!!!NO INPUT MESSAGE RECEIVED - video replayer might not be sending data");
        return;
    }
    
    HOLOSCAN_LOG_INFO("!!!!INPUT MESSAGE RECEIVED - video replayer is sending data");
    
    // Get tensor from input message
    auto tensor = input_message.value().get<holoscan::Tensor>();
    if (!tensor) {
        HOLOSCAN_LOG_ERROR("!!!!NO TENSOR IN INPUT MESSAGE - format converter issue");
        return;
    }
    
    HOLOSCAN_LOG_INFO("!!!!TENSOR RECEIVED from video pipeline - checking tensor properties");

    HOLOSCAN_LOG_DEBUG("StreamingClientOp::compute() called - send_frames: {}, receive_frames: {}", 
                     send_frames_.get(), receive_frames_.get());
  
  // Process input frames if we're sending
  if (send_frames_.get()) {
    HOLOSCAN_LOG_DEBUG("Processing input frames for sending...");
    
    // Check frame rate limiting
    auto current_time = std::chrono::steady_clock::now();
    auto time_since_last_frame = current_time - last_frame_time_;
    
    if (time_since_last_frame < frame_interval_) {
      // Skip this frame to maintain target frame rate
      HOLOSCAN_LOG_DEBUG("Skipping frame to maintain frame rate ({}ms since last, interval={}ms)", 
                         std::chrono::duration_cast<std::chrono::milliseconds>(time_since_last_frame).count(),
                         std::chrono::duration_cast<std::chrono::milliseconds>(frame_interval_).count());
      return;
    }
    
    last_frame_time_ = current_time;
    HOLOSCAN_LOG_DEBUG("Frame rate check passed, processing frame...");
    
    auto entity = input_message.value();
    HOLOSCAN_LOG_DEBUG("Successfully received input entity");

    // Validate tensor - accept both 1D (encoded) and 3D (raw image) tensors
    if ((tensor->ndim() != 1 && tensor->ndim() != 3) || tensor->size() == 0) {
        HOLOSCAN_LOG_ERROR("Invalid tensor: ndim={}, size={}. Expected 1D (encoded) or 3D (HWC image) tensor.", 
                           tensor->ndim(), tensor->size());
        return;
    }

    // Log tensor information for debugging
    HOLOSCAN_LOG_INFO("Processing tensor: ndim={}, size={}, nbytes={}, device={}",
                       tensor->ndim(), tensor->size(), tensor->nbytes(),
                       tensor->device().device_type == kDLCUDA ? "GPU" : "CPU");

    // Log tensor shape for debugging
    auto shape = tensor->shape();
    std::string shape_str = "shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) shape_str += ", ";
        shape_str += std::to_string(shape[i]);
    }
    shape_str += "]";
    HOLOSCAN_LOG_INFO("Tensor {}", shape_str);

    // Handle different tensor formats
    bool is_raw_image = (tensor->ndim() == 3);
    size_t expected_size;
    PixelFormat detected_format = PixelFormat::BGR;
    
    if (is_raw_image) {
      // For 3D tensor (HWC), calculate expected size for BGR format
      auto shape = tensor->shape();
      if (shape.size() >= 3) {
        size_t height = shape[0];
        size_t width = shape[1];
        size_t channels = shape[2];
        
        // Check tensor data type
        auto dtype = tensor->dtype();
        bool is_float32 = (dtype.code == kDLFloat);
        
        if (is_float32) {
          HOLOSCAN_LOG_INFO("Processing float32 tensor: {}x{}x{} channels", height, width, channels);
          // For float32, values are in range 0-255, we need to convert to uint8
          expected_size = height * width * 3; // BGR: 3 channels, uint8 output
        } else {
          HOLOSCAN_LOG_INFO("Processing uint8 tensor: {}x{}x{} channels", height, width, channels);
          expected_size = height * width * 3; // BGR: 3 channels
        }
        
        detected_format = PixelFormat::BGR;
      } else {
        expected_size = tensor->nbytes();
      }
    } else {
      // For other tensors, use the actual tensor size
      expected_size = tensor->nbytes();
      HOLOSCAN_LOG_DEBUG("Non-raw tensor: using actual size={}", expected_size);
    }

    // Create a video frame from the tensor data
    // Simplified approach: just pass through the uint8 data from format converter
    size_t actual_size = tensor->nbytes();
    
    HOLOSCAN_LOG_INFO("SIMPLIFIED TENSOR PROCESSING:");
    HOLOSCAN_LOG_INFO("  - Tensor data pointer: {}", tensor->data() ? "VALID" : "NULL");
    HOLOSCAN_LOG_INFO("  - Tensor size: {} bytes", actual_size);
    HOLOSCAN_LOG_INFO("  - Tensor shape: {}", tensor->shape());
    HOLOSCAN_LOG_INFO("  - Tensor dtype: {}", tensor->dtype().code);
    
    VideoFrame frame(width_.get(), height_.get());
    frame.setFormat(PixelFormat::BGR);
    
    // Safely copy data with bounds checking
    if (tensor->data() && actual_size > 0) {
      uint8_t* frame_data = frame.getWritableData();
      size_t frame_buffer_size = frame.getDataSize();
      
      HOLOSCAN_LOG_INFO("Frame buffer size: {}, tensor size: {}", frame_buffer_size, actual_size);
      
      if (frame_data && frame_buffer_size >= actual_size) {
        // Check if tensor is on GPU or CPU
        auto tensor_device = tensor->device();
        
        if (tensor_device.device_type == kDLCUDA) {
          // GPU memory - simple direct copy
          HOLOSCAN_LOG_INFO("Copying from GPU tensor to frame buffer");
          cudaError_t err = cudaMemcpy(frame_data, tensor->data(), actual_size, cudaMemcpyDeviceToHost);
          if (err != cudaSuccess) {
              HOLOSCAN_LOG_ERROR("CUDA memcpy failed: {}", cudaGetErrorString(err));
              return;
          }
        } else {
          // CPU memory - simple direct copy
          HOLOSCAN_LOG_INFO("Copying from CPU tensor to frame buffer");
          std::memcpy(frame_data, tensor->data(), actual_size);
        }
        
        HOLOSCAN_LOG_INFO("✅ Simplified tensor processing completed: {} bytes copied", actual_size);
      } else {
        HOLOSCAN_LOG_ERROR("Frame buffer too small: buffer={}, needed={}", frame_buffer_size, actual_size);
        return;
      }
    }
    
    // Set proper timestamp instead of 0
    frame.setTimestamp(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count());
    // Set pixel format based on detected format
    frame.setFormat(detected_format);
    
    // Get the actual frame size that will be sent
    size_t actual_frame_size = frame.getDataSize();
                  
    // Send the frame via the streaming client
    if (client_) {
      // Add comprehensive frame debugging before sending
      HOLOSCAN_LOG_INFO("!!!!CLIENT FRAME SENDING DEBUG:");
      HOLOSCAN_LOG_INFO("  - Frame dimensions: {}x{}", frame.getWidth(), frame.getHeight());
      HOLOSCAN_LOG_INFO("  - Frame format: {}", static_cast<int>(frame.getFormat()));
      HOLOSCAN_LOG_INFO("  - Frame data size: {} bytes", frame.getDataSize());
      HOLOSCAN_LOG_INFO("  - Frame timestamp: {}", frame.getTimestamp());
      HOLOSCAN_LOG_INFO("  - Frame data pointer: {}", frame.getData() ? "VALID" : "NULL");
      
      // Check if frame data is valid
      const uint8_t* frame_data = frame.getData();
      size_t data_size = frame.getDataSize();
      
      if (!frame_data || data_size == 0) {
        HOLOSCAN_LOG_ERROR("!!!!CLIENT FRAME VALIDATION FAILED:");
        HOLOSCAN_LOG_ERROR("  - Frame data pointer: {}", frame_data ? "VALID" : "NULL");
        HOLOSCAN_LOG_ERROR("  - Frame data size: {} bytes", data_size);
        HOLOSCAN_LOG_ERROR("  - This frame will likely arrive empty at the server!");
        return;
      }
      
      // Check if frame data is all zeros
      bool all_zeros = true;
      size_t non_zero_count = 0;
      size_t check_limit = std::min(data_size, static_cast<size_t>(100)); // Check first 100 bytes
      
      for (size_t i = 0; i < check_limit; ++i) {
        if (frame_data[i] != 0) {
          all_zeros = false;
          non_zero_count++;
        }
      }
      
      HOLOSCAN_LOG_INFO("  - First {} bytes analysis: all_zeros={}, non_zero_count={}", 
                       check_limit, all_zeros ? "YES" : "NO", non_zero_count);
      
      // Log first few bytes as hex
      if (data_size >= 16) {
        std::string hex_dump;
        for (size_t i = 0; i < 16; ++i) {
          char hex_byte[4];
          snprintf(hex_byte, sizeof(hex_byte), "%02X ", frame_data[i]);
          hex_dump += hex_byte;
        }
        HOLOSCAN_LOG_INFO("  - First 16 bytes (hex): {}", hex_dump);
      }
      
      // Check if dimensions match expected values
      if (frame.getWidth() != width_.get() || frame.getHeight() != height_.get()) {
        HOLOSCAN_LOG_WARN("!!!!CLIENT FRAME DIMENSION MISMATCH:");
        HOLOSCAN_LOG_WARN("  - Frame dimensions: {}x{}", frame.getWidth(), frame.getHeight());
        HOLOSCAN_LOG_WARN("  - Expected dimensions: {}x{}", width_.get(), height_.get());
      }
      
      // Check if frame size makes sense for the dimensions
      size_t expected_size = frame.getWidth() * frame.getHeight() * 3; // BGR = 3 channels
      if (data_size < expected_size) {
        HOLOSCAN_LOG_ERROR("!!!!CLIENT FRAME SIZE VALIDATION FAILED:");
        HOLOSCAN_LOG_ERROR("  - Frame data size: {} bytes", data_size);
        HOLOSCAN_LOG_ERROR("  - Expected minimum: {} bytes (for BGR)", expected_size);
        HOLOSCAN_LOG_ERROR("  - This suggests incomplete frame data creation");
        return;
      }
      
      HOLOSCAN_LOG_INFO("!!!!CLIENT FRAME VALIDATION PASSED - sending to server");
      
      HOLOSCAN_LOG_INFO("Attempting to send frame to server: {}x{}, format=BGR, frame_size: {} bytes (original tensor: {} bytes)", 
                        frame.getWidth(), frame.getHeight(), 
                        actual_frame_size, tensor->nbytes());
      
      // Send the frame
      client_->sendFrame(frame);
      
      HOLOSCAN_LOG_INFO("✅ Successfully sent BGR frame to server, frame_size: {} bytes", actual_frame_size);
    } else {
      HOLOSCAN_LOG_ERROR("StreamingClient is null - cannot send frame!");
    }
  } else {
    HOLOSCAN_LOG_WARN("🚨 CRITICAL: Frame sending is DISABLED (send_frames=false) - This is why server receives no frames!");
  }

  // Output received frames if we have any
  if (receive_frames_.get()) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (has_new_frame_) {
      try {
        auto& frame_ref = current_frame_; // Reference to avoid copying
        
        // Get frame data
        const uint8_t* frame_data = frame_ref.getData();
        size_t data_size = frame_ref.getDataSize();
        
        if (!frame_data || data_size == 0) {
          HOLOSCAN_LOG_ERROR("Invalid frame data");
          return;
        }
        
        // Create a GXF entity
        auto maybe_entity = nvidia::gxf::Entity::New(context.context());
        if (!maybe_entity) {
          HOLOSCAN_LOG_ERROR("Failed to create entity");
          return;
        }
        
        // Add a tensor to the entity
        auto tensor_handle = maybe_entity.value().add<nvidia::gxf::Tensor>("frame_tensor");
        if (!tensor_handle) {
          HOLOSCAN_LOG_ERROR("Failed to add tensor to entity");
          return;
        }
        
        // Set up dimensions and shape for BGR format
        nvidia::gxf::Shape shape = nvidia::gxf::Shape{
          static_cast<int32_t>(frame_ref.getHeight()),
          static_cast<int32_t>(frame_ref.getWidth()),
          3  // BGR has 3 channels
        };
        
        // Set up element type
        nvidia::gxf::PrimitiveType element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
        int element_size = nvidia::gxf::PrimitiveTypeSize(element_type);
        
        // Create a copy of the frame data in host memory
        auto host_data = std::shared_ptr<uint8_t[]>(new uint8_t[data_size]);
        if (!host_data) {
          HOLOSCAN_LOG_ERROR("Failed to allocate memory for frame data");
          return;
        }
        
        // Copy the frame data
        std::memcpy(host_data.get(), frame_data, data_size);
        
        // Wrap the memory in the tensor
        tensor_handle.value()->wrapMemory(
            shape,
            element_type,
            element_size,
            nvidia::gxf::ComputeTrivialStrides(shape, element_size),
            nvidia::gxf::MemoryStorageType::kSystem,
            host_data.get(),
            [host_data](void*) mutable {
              // The shared_ptr will automatically clean up when it goes out of scope
              host_data.reset();
              return nvidia::gxf::Success;
            });
        
        // Emit the entity
        op_output.emit(maybe_entity.value(), "output_frames");
        HOLOSCAN_LOG_DEBUG("Emitted frame: {}x{}", frame_ref.getWidth(), frame_ref.getHeight());
        
        has_new_frame_ = false;  // Reset flag after emitting
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Error in receive_frames processing: {}", e.what());
      }
    }
  }
}

void StreamingClientOp::onFrameReceived(const VideoFrame& frame) {
  std::lock_guard<std::mutex> lock(frame_mutex_);
  
  // Store the new frame
  current_frame_ = frame;
  has_new_frame_ = true;
  
  HOLOSCAN_LOG_DEBUG("Received frame: {}x{}", frame.getWidth(), frame.getHeight());
}

VideoFrame StreamingClientOp::generateFrame() {
  std::lock_guard<std::mutex> lock(frame_mutex_);
  
  // Create a simple dummy frame
  // In a real application, this would use input frames from the compute method
  return VideoFrame(width_.get(), height_.get(), nullptr, 0, 0);
}

} // namespace holoscan::ops





