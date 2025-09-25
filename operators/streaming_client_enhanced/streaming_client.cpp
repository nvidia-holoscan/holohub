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
#include <chrono>
#include <iomanip>
#include <sstream>
#include <fstream>

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

// Utility function to write video frame to disk for debugging
bool writeFrameToDisk(const VideoFrame& frame, const std::string& filename_prefix, int frame_number = -1) {
    try {
        // Generate filename with timestamp and frame number
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream filename;
        filename << filename_prefix;
        if (frame_number >= 0) {
            filename << "_frame" << std::setfill('0') << std::setw(6) << frame_number;
        }
        filename << "_" << std::put_time(std::localtime(&time_t), "%H%M%S")
                 << "_" << std::setfill('0') << std::setw(3) << ms.count();

        // Get frame properties
        uint32_t width = frame.getWidth();
        uint32_t height = frame.getHeight();
        const uint8_t* data = frame.getData();
        size_t data_size = frame.getDataSize();
        PixelFormat format = frame.getFormat();

        // 🔍 ENHANCED VALIDATION: Add detailed frame analysis
        HOLOSCAN_LOG_INFO("🔍 writeFrameToDisk DEBUG for {}:", filename_prefix);
        HOLOSCAN_LOG_INFO("  - Width: {}", width);
        HOLOSCAN_LOG_INFO("  - Height: {}", height);
        HOLOSCAN_LOG_INFO("  - Data pointer: {}", static_cast<const void*>(data));
        HOLOSCAN_LOG_INFO("  - Data size: {} bytes", data_size);
        HOLOSCAN_LOG_INFO("  - Format: {} ({})", static_cast<int>(format),
                         (format == PixelFormat::BGRA) ? "BGRA" :
                         (format == PixelFormat::BGR) ? "BGR" :
                         (format == PixelFormat::RGBA) ? "RGBA" : "Unknown");

        if (!data || data_size == 0 || width == 0 || height == 0) {
            HOLOSCAN_LOG_ERROR("❌ writeFrameToDisk: Invalid frame data - width={}, height={}, data={}, size={}",
                               width, height, static_cast<const void*>(data), data_size);
            return false;
        }

        // 🔍 MEMORY ANALYSIS: Check if data appears to be valid
        if (data && data_size >= 20) {
            HOLOSCAN_LOG_INFO("  - Frame data first 20 bytes: {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {} {}",
                             data[0], data[1], data[2], data[3], data[4],
                             data[5], data[6], data[7], data[8], data[9],
                             data[10], data[11], data[12], data[13], data[14],
                             data[15], data[16], data[17], data[18], data[19]);

            // Check if all data is zeros (potential issue)
            bool all_zeros = true;
            size_t check_bytes = std::min(data_size, static_cast<size_t>(100));
            for (size_t i = 0; i < check_bytes; ++i) {
                if (data[i] != 0) {
                    all_zeros = false;
                    break;
                }
            }

            if (all_zeros) {
                HOLOSCAN_LOG_WARN("⚠️  WARNING: First {} bytes of frame data are all zeros!", check_bytes);
            } else {
                HOLOSCAN_LOG_INFO("✅ Frame data appears valid (contains non-zero bytes)");
            }
        }

        // Write raw binary data
        std::string raw_filename = filename.str() + "_raw.bin";
        std::ofstream raw_file(raw_filename, std::ios::binary);
        if (raw_file.is_open()) {
            raw_file.write(reinterpret_cast<const char*>(data), data_size);
            raw_file.close();
            HOLOSCAN_LOG_INFO("Wrote raw frame data to: {}", raw_filename);
        }

        // Write metadata file
        std::string meta_filename = filename.str() + "_meta.txt";
        std::ofstream meta_file(meta_filename);
        if (meta_file.is_open()) {
            meta_file << "Frame Metadata:\n";
            meta_file << "Width: " << width << "\n";
            meta_file << "Height: " << height << "\n";
            meta_file << "Data Size: " << data_size << " bytes\n";
            meta_file << "Pixel Format: " << static_cast<int>(format) << " (";
            switch (format) {
                case PixelFormat::BGR: meta_file << "BGR"; break;
                case PixelFormat::BGRA: meta_file << "BGRA"; break;
                case PixelFormat::RGBA: meta_file << "RGBA"; break;
                default: meta_file << "Unknown"; break;
            }
            meta_file << ")\n";
            meta_file << "Timestamp: " << frame.getTimestamp() << "\n";
            meta_file << "Bytes per pixel: " << (data_size / (width * height)) << "\n";

            // Add first few pixel values for inspection
            meta_file << "\nFirst 10 pixels (raw bytes):\n";
            size_t bytes_per_pixel = (format == PixelFormat::BGRA || format == PixelFormat::RGBA) ? 4 : 3;
            for (int i = 0; i < std::min(10, static_cast<int>(width * height)) && i * bytes_per_pixel < data_size; ++i) {
                meta_file << "Pixel " << i << ": ";
                for (size_t j = 0; j < bytes_per_pixel && (i * bytes_per_pixel + j) < data_size; ++j) {
                    meta_file << static_cast<int>(data[i * bytes_per_pixel + j]) << " ";
                }
                meta_file << "\n";
            }

            meta_file.close();
            HOLOSCAN_LOG_INFO("Wrote frame metadata to: {}", meta_filename);
        }

        // Write as PPM image file (for easy viewing)
        if (format == PixelFormat::BGRA || format == PixelFormat::BGR || format == PixelFormat::RGBA) {
            std::string ppm_filename = filename.str() + ".ppm";
            std::ofstream ppm_file(ppm_filename, std::ios::binary);
            if (ppm_file.is_open()) {
                // PPM header
                ppm_file << "P6\n" << width << " " << height << "\n255\n";

                // Convert pixel data to RGB for PPM
                size_t bytes_per_pixel = (format == PixelFormat::BGRA || format == PixelFormat::RGBA) ? 4 : 3;
                for (uint32_t y = 0; y < height; ++y) {
                    for (uint32_t x = 0; x < width; ++x) {
                        size_t pixel_offset = (y * width + x) * bytes_per_pixel;
                        if (pixel_offset + 2 < data_size) {
                            uint8_t r, g, b;
                            if (format == PixelFormat::BGRA || format == PixelFormat::BGR) {
                                // BGR(A) format - swap B and R
                                b = data[pixel_offset + 0];
                                g = data[pixel_offset + 1];
                                r = data[pixel_offset + 2];
                            } else {
                                // RGB(A) format
                                r = data[pixel_offset + 0];
                                g = data[pixel_offset + 1];
                                b = data[pixel_offset + 2];
                            }
                            ppm_file.write(reinterpret_cast<const char*>(&r), 1);
                            ppm_file.write(reinterpret_cast<const char*>(&g), 1);
                            ppm_file.write(reinterpret_cast<const char*>(&b), 1);
                        }
                    }
                }
                ppm_file.close();
                HOLOSCAN_LOG_INFO("Wrote PPM image to: {} (can be viewed with image viewers)", ppm_filename);
            }
        }

        return true;

    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("writeFrameToDisk exception: {}", e.what());
        return false;
    }
}

// Utility function to write tensor data to disk for debugging
bool writeTensorToDisk(const std::shared_ptr<holoscan::Tensor>& tensor, const std::string& filename_prefix, int frame_number = -1) {
    try {
        if (!tensor || !tensor->data()) {
            HOLOSCAN_LOG_ERROR("writeTensorToDisk: Invalid tensor");
            return false;
        }

        // Generate filename
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream filename;
        filename << filename_prefix;
        if (frame_number >= 0) {
            filename << "_frame" << std::setfill('0') << std::setw(6) << frame_number;
        }
        filename << "_" << std::put_time(std::localtime(&time_t), "%H%M%S")
                 << "_" << std::setfill('0') << std::setw(3) << ms.count();

        auto shape = tensor->shape();
        if (shape.size() != 3) {
            HOLOSCAN_LOG_ERROR("writeTensorToDisk: Expected 3D tensor");
            return false;
        }

        int height = static_cast<int>(shape[0]);
        int width = static_cast<int>(shape[1]);
        int channels = static_cast<int>(shape[2]);

        // Write tensor metadata
        std::string meta_filename = filename.str() + "_tensor_meta.txt";
        std::ofstream meta_file(meta_filename);
        if (meta_file.is_open()) {
            meta_file << "Tensor Metadata:\n";
            meta_file << "Shape: [" << height << ", " << width << ", " << channels << "]\n";
            meta_file << "Data Type: code=" << static_cast<int>(tensor->dtype().code)
                     << ", bits=" << tensor->dtype().bits << "\n";
            meta_file << "Device: " << (tensor->device().device_type == kDLCUDA ? "GPU" : "CPU") << "\n";
            meta_file << "Size: " << tensor->nbytes() << " bytes\n";
            meta_file.close();
        }

        // Copy tensor data to host if on GPU
        std::vector<uint8_t> host_data;
        const uint8_t* data_ptr = nullptr;

        if (tensor->device().device_type == kDLCUDA) {
            host_data.resize(tensor->nbytes());
            CUDA_TRY(cudaMemcpy(host_data.data(), tensor->data(), tensor->nbytes(), cudaMemcpyDeviceToHost));
            data_ptr = host_data.data();
        } else {
            data_ptr = static_cast<const uint8_t*>(tensor->data());
        }

        // Write raw tensor data
        std::string raw_filename = filename.str() + "_tensor_raw.bin";
        std::ofstream raw_file(raw_filename, std::ios::binary);
        if (raw_file.is_open()) {
            raw_file.write(reinterpret_cast<const char*>(data_ptr), tensor->nbytes());
            raw_file.close();
            HOLOSCAN_LOG_INFO("Wrote raw tensor data to: {}", raw_filename);
        }

        // Write as PPM if it's BGR/RGB data
        if (channels == 3) {
            std::string ppm_filename = filename.str() + "_tensor.ppm";
            std::ofstream ppm_file(ppm_filename, std::ios::binary);
            if (ppm_file.is_open()) {
                // PPM header
                ppm_file << "P6\n" << width << " " << height << "\n255\n";

                // Assume BGR format and convert to RGB for PPM
                for (int y = 0; y < height; ++y) {
                    for (int x = 0; x < width; ++x) {
                        size_t pixel_offset = (y * width + x) * 3;
                        if (pixel_offset + 2 < tensor->nbytes()) {
                            // BGR to RGB conversion
                            uint8_t b = data_ptr[pixel_offset + 0];
                            uint8_t g = data_ptr[pixel_offset + 1];
                            uint8_t r = data_ptr[pixel_offset + 2];
                            ppm_file.write(reinterpret_cast<const char*>(&r), 1);
                            ppm_file.write(reinterpret_cast<const char*>(&g), 1);
                            ppm_file.write(reinterpret_cast<const char*>(&b), 1);
                        }
                    }
                }
                ppm_file.close();
                HOLOSCAN_LOG_INFO("Wrote tensor PPM image to: {}", ppm_filename);
            }
        }

        return true;

    } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("writeTensorToDisk exception: {}", e.what());
        return false;
    }
}

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
             "Port used for signaling", uint16_t{48010});  // Match Holoscan Streaming Stack hardcoded port
  spec.param(receive_frames_, "receive_frames", "Receive Frames",
             "Whether to receive frames from server", true);
  spec.param(send_frames_, "send_frames", "Send Frames",
             "Whether to send frames to server", true);
  spec.param(min_non_zero_bytes_, "min_non_zero_bytes", "Minimum Non-Zero Bytes",
             "Minimum number of non-zero bytes required to consider a frame valid (prevents sending empty frames)",
             100u);

  // Print the parameters for debugging with correct values
  HOLOSCAN_LOG_INFO("StreamingClientOp setup with defaults: width={}, height={}, fps={}, server_ip={}, port={}, send_frames={}, min_non_zero_bytes={}",
                    854u, 480u, 30u, "127.0.0.1", 48010, true, 100u);
}

void StreamingClientOp::initialize() {
  // CRITICAL: Set environment variables BEFORE calling Operator::initialize()
  // Use comprehensive logging configuration for Holoscan Streaming Stack team debugging

  // Enable DEBUG level logging and comprehensive debugging for Holoscan Streaming Stack team
  setenv("NVST_LOG_LEVEL", "DEBUG", 1);  // Enable DEBUG level for comprehensive debugging
  setenv("NVST_ALLOW_SELF_SIGNED_CERTS", "1", 1);
  setenv("NVST_SKIP_CERTIFICATE_VALIDATION", "1", 1);

  HOLOSCAN_LOG_INFO("Set NVIDIA streaming environment variables with DEBUG logging enabled");

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

  HOLOSCAN_LOG_INFO("Initializing StreamingClient with parameters:");
  HOLOSCAN_LOG_INFO("  - Width: {}", width_.get());
  HOLOSCAN_LOG_INFO("  - Height: {}", height_.get());
  HOLOSCAN_LOG_INFO("  - FPS: {}", fps_.get());
  HOLOSCAN_LOG_INFO("  - Server IP: {}", server_ip_.get());
  HOLOSCAN_LOG_INFO("  - Signaling Port: {}", signaling_port_.get());

  // Initialize timing control
  frame_interval_ = std::chrono::microseconds(1000000 / fps_.get());  // Convert fps to microseconds
  last_frame_time_ = std::chrono::steady_clock::now();

  // Initialize client with the validated parameters
  client_ = std::make_unique<StreamingClient>(
      width_.get(), height_.get(), fps_.get(), signaling_port_.get());

  // Log client creation success
  if (client_) {
    HOLOSCAN_LOG_INFO("🔧 ENHANCED StreamingClient constructed! Version with buffer validation fixes!");
    HOLOSCAN_LOG_INFO("StreamingClient created successfully");
  } else {
    HOLOSCAN_LOG_ERROR("Failed to create StreamingClient");
  }

  // REMOVED: Frame source callback that returned empty frames
  // We use sendFrame() directly in compute() instead
  // This eliminates the conflict between frame source callback and direct sending

  HOLOSCAN_LOG_INFO("StreamingClientOp initialized successfully");
}

void StreamingClientOp::start() {
  if (!client_) {
    HOLOSCAN_LOG_ERROR("Cannot start streaming: client not initialized");
    return;
  }

  try {
    // Add more comprehensive connection setup
    HOLOSCAN_LOG_INFO("=== STARTING STREAMING CLIENT ===");
    HOLOSCAN_LOG_INFO("Target server: {}:{}", server_ip_.get(), signaling_port_.get());
    HOLOSCAN_LOG_INFO("Client configuration:");
    HOLOSCAN_LOG_INFO("  - Resolution: {}x{}", width_.get(), height_.get());
    HOLOSCAN_LOG_INFO("  - FPS: {}", fps_.get());
    HOLOSCAN_LOG_INFO("  - Send frames: {}", send_frames_.get());
    HOLOSCAN_LOG_INFO("  - Receive frames: {}", receive_frames_.get());

    // Set frame callback BEFORE starting streaming
    if (receive_frames_.get()) {
      HOLOSCAN_LOG_INFO("Setting up frame received callback...");
      client_->setFrameReceivedCallback([this](const VideoFrame& frame) {
        onFrameReceived(frame);
      });
      HOLOSCAN_LOG_INFO("✅ Frame callback configured");
    }

    // Add pre-connection validation
    if (server_ip_.get().empty() || signaling_port_.get() == 0) {
      throw std::runtime_error("Invalid server configuration");
    }

    // Add server connectivity test
    HOLOSCAN_LOG_INFO("Testing server connectivity to {}:{}...", server_ip_.get(), signaling_port_.get());
    // Note: A full socket test could be added here, but for now we'll rely on the Holoscan Streaming Stack connection attempt

    // Implement robust connection with exponential backoff
    int max_retries = 3;  // Reduced from 5
    int retry_count = 0;
    bool connection_successful = false;

    while (retry_count < max_retries && !connection_successful) {
      try {
        HOLOSCAN_LOG_INFO("🔄 Connection attempt {} of {}", retry_count + 1, max_retries);

        // Add delay between retries (exponential backoff)
        if (retry_count > 0) {
          int delay_ms = 1000 * retry_count;  // 1s, 2s, 3s (simplified)
          HOLOSCAN_LOG_INFO("Waiting {}ms before retry...", delay_ms);
          std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
        }

        client_->startStreaming(server_ip_.get(), signaling_port_.get());

        // CRITICAL: Check if actually connected
        if (!client_->isStreaming()) {
          throw std::runtime_error("startStreaming returned but client is not streaming");
        }

        connection_successful = true;
        HOLOSCAN_LOG_INFO("✅ Connection established successfully");

        // Add immediate status check
        HOLOSCAN_LOG_INFO("🔍 POST-CONNECTION STATUS:");
        HOLOSCAN_LOG_INFO("  - Client streaming: {}", client_->isStreaming() ? "YES" : "NO");
        HOLOSCAN_LOG_INFO("  - Upstream ready: {}", client_->isUpstreamReady() ? "YES" : "NO");

        // Wait for upstream connection to be established
        HOLOSCAN_LOG_INFO("⏳ Waiting for upstream connection to be ready...");

        // Wait up to 2 seconds for connection to stabilize (reduced from 3)
        auto start_wait = std::chrono::steady_clock::now();
        const auto upstream_timeout = std::chrono::seconds(2);

        int wait_attempts = 0;
        // Check for upstream readiness using the actual method
        while ((std::chrono::steady_clock::now() - start_wait) < upstream_timeout && !client_->isUpstreamReady()) {
          wait_attempts++;
          if (wait_attempts % 10 == 0) { // Log every second (10 * 100ms)
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
              std::chrono::steady_clock::now() - start_wait).count();
            HOLOSCAN_LOG_INFO("⏳ Waiting for upstream to be ready... attempt {}, elapsed {}ms",
                             wait_attempts, elapsed);
            HOLOSCAN_LOG_INFO("  - Client streaming: {}", client_->isStreaming() ? "YES" : "NO");
            HOLOSCAN_LOG_INFO("  - Upstream ready: {}", client_->isUpstreamReady() ? "YES" : "NO");
          }
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        // Report final status after wait
        if (client_->isUpstreamReady()) {
          HOLOSCAN_LOG_INFO("✅ Upstream connection established successfully!");
        } else {
          HOLOSCAN_LOG_WARN("⚠️ Upstream connection not ready after timeout - will continue anyway");
        }

      } catch (const std::exception& e) {
        retry_count++;
        HOLOSCAN_LOG_WARN("❌ Connection attempt {} failed: {}", retry_count, e.what());

        // Log connection details but don't be too verbose
        if (retry_count == 1) { // Only log details on first failure
          HOLOSCAN_LOG_INFO("Connection details: server={}:{}", server_ip_.get(), signaling_port_.get());
        }

        if (retry_count >= max_retries) {
          HOLOSCAN_LOG_ERROR("❌ All connection attempts failed. Final error: {}", e.what());
          // Don't throw - let compute() handle retries
          return;
        }
      }
    }

    // Add post-connection validation
    if (connection_successful) {
      // Wait for connection to stabilize
      std::this_thread::sleep_for(std::chrono::milliseconds(500));

      // Verify streaming state
      if (client_->isStreaming()) {
        HOLOSCAN_LOG_INFO("✅ Client is streaming and ready");
        // Reset retry counter on successful connection
        retry_count_ = 0;
        last_retry_time_ = std::chrono::steady_clock::now();
      } else {
        HOLOSCAN_LOG_WARN("⚠️ Client connected but not streaming");
      }

      // Wait for first frame if receiving
      if (receive_frames_.get()) {
        HOLOSCAN_LOG_INFO("⏳ Waiting for first frame from server...");
        // Wait briefly to see if frames start arriving
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        std::lock_guard<std::mutex> lock(frame_mutex_);
        if (has_new_frame_) {
          HOLOSCAN_LOG_INFO("✅ First frame received from server");
        } else {
          HOLOSCAN_LOG_WARN("⚠️ No frame received yet (may be normal for server startup)");
        }
      }
    }

  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("❌ Failed to start streaming client: {}", e.what());
    // Don't throw - let compute() handle the error state
  }
}

void StreamingClientOp::stop() {
  if (client_) {
    try {
      HOLOSCAN_LOG_INFO("Stopping StreamingClient...");

      // Clear frame callback to prevent race conditions
      if (receive_frames_.get()) {
        try {
          client_->setFrameReceivedCallback(nullptr);
        } catch (const std::exception& e) {
          HOLOSCAN_LOG_WARN("Exception clearing frame callback: {}", e.what());
        }
      }

      // Stop streaming with timeout and better error handling
      try {
        client_->stopStreaming();
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_WARN("Exception during stopStreaming: {}", e.what());
        // Continue with cleanup even if stop fails
      }

      // Wait for streaming to end with timeout
      auto stop_timeout = std::chrono::steady_clock::now() + std::chrono::seconds(3);  // Reduced timeout
      bool stopped = false;

      while (!stopped && std::chrono::steady_clock::now() < stop_timeout) {
        try {
          stopped = client_->waitForStreamingEnded();
          if (!stopped) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));  // Shorter sleep
          }
        } catch (const std::exception& e) {
          HOLOSCAN_LOG_WARN("Exception while waiting for streaming to end: {}", e.what());
          // Don't break - give it one more chance
          std::this_thread::sleep_for(std::chrono::milliseconds(100));
          break;
        }
      }

      if (!stopped) {
        HOLOSCAN_LOG_WARN("StreamingClient did not stop gracefully within timeout");
        // Don't force anything - just proceed with cleanup
      } else {
        HOLOSCAN_LOG_INFO("StreamingClient stopped gracefully");
      }

    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Error stopping StreamingClient: {}", e.what());
    } catch (...) {
      HOLOSCAN_LOG_ERROR("Unknown error stopping StreamingClient");
    }

    // SAFER: Don't immediately reset client pointer
    // Let it go out of scope naturally to avoid potential cleanup issues
    try {
      // Add a small delay to let any background threads finish
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
      client_.reset();
      HOLOSCAN_LOG_INFO("StreamingClient cleaned up successfully");
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Exception during client cleanup: {}", e.what());
    } catch (...) {
      HOLOSCAN_LOG_ERROR("Unknown exception during client cleanup");
    }
  }
}

void StreamingClientOp::compute(holoscan::InputContext& op_input,
                             holoscan::OutputContext& op_output,
                             holoscan::ExecutionContext& context) {

  // Add detailed connection state logging
  static int compute_call_count = 0;
  static int connection_retry_interval = 0;
  static int debug_frame_counter = 0;  // Add debug frame counter
  compute_call_count++;

  if (compute_call_count % 30 == 0) { // Log every second at 30fps
    HOLOSCAN_LOG_INFO("Compute called {} times, client streaming: {}",
                     compute_call_count, client_ ? (client_->isStreaming() ? "true" : "false") : "null");
  }

  // Check if client exists
  if (!client_) {
    if (compute_call_count % 60 == 0) {
      HOLOSCAN_LOG_ERROR("Client is null - this should not happen");
    }
    return;
  }

  // If not streaming, try to reconnect periodically
  if (!client_->isStreaming()) {
    connection_retry_interval++;

    // Retry every 5 seconds (150 frames at 30fps)
    if (connection_retry_interval >= 150) {
      connection_retry_interval = 0;

      HOLOSCAN_LOG_WARN("Client not streaming, attempting to reconnect...");

      // SAFER APPROACH: Track reconnection attempts but avoid aggressive client recreation
      static int reconnection_attempts = 0;
      static auto last_reconnect_time = std::chrono::steady_clock::now();

      auto now = std::chrono::steady_clock::now();
      auto time_since_last_reconnect = std::chrono::duration_cast<std::chrono::seconds>(now - last_reconnect_time).count();

      try {
        // Only attempt reconnection if we haven't tried too recently
        if (time_since_last_reconnect >= 5) { // Wait at least 5 seconds between attempts
          reconnection_attempts++;
          last_reconnect_time = now;

          HOLOSCAN_LOG_INFO("🔄 Reconnection attempt {} (last attempt {} seconds ago)",
                           reconnection_attempts, time_since_last_reconnect);

          // REMOVED: Aggressive client recreation that caused segfaults
          // Instead, just try to reconnect with existing client
          client_->startStreaming(server_ip_.get(), signaling_port_.get());

          // Check if reconnection was successful
          std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Brief wait
          if (client_->isStreaming()) {
            HOLOSCAN_LOG_INFO("✅ Reconnection successful!");
            reconnection_attempts = 0;  // Reset on success
          } else {
            HOLOSCAN_LOG_WARN("❌ Reconnection attempt {} failed - client not streaming", reconnection_attempts);

            // If we've tried many times, give a longer break
            if (reconnection_attempts >= 10) {
              HOLOSCAN_LOG_WARN("⏸️  Too many reconnection attempts ({}), will wait longer before trying again", reconnection_attempts);
              reconnection_attempts = 0;  // Reset counter to avoid infinite accumulation
              last_reconnect_time = now + std::chrono::seconds(30);  // Wait 30 more seconds
            }
          }
        } else {
          HOLOSCAN_LOG_DEBUG("Skipping reconnection attempt (only {} seconds since last attempt)", time_since_last_reconnect);
        }

      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("❌ Reconnection failed: {}", e.what());

        // Log specific error types for debugging but don't force client recreation
        std::string error_msg = e.what();
        if (error_msg.find("NVST_R_INVALID_OPERATION") != std::string::npos) {
          HOLOSCAN_LOG_WARN("⚠️  DETECTED: NVST_R_INVALID_OPERATION - Client may be in error state");
          HOLOSCAN_LOG_WARN("   Will continue attempting reconnection without forcing client recreation");
        }

        // Give it more time before next attempt if we get specific errors
        if (error_msg.find("STATE_ERROR") != std::string::npos ||
            error_msg.find("NVST_R_INVALID_OPERATION") != std::string::npos) {
          last_reconnect_time = now + std::chrono::seconds(10);  // Wait 10 seconds for state errors
        }
      }
    }

    if (compute_call_count % 60 == 0) {
      HOLOSCAN_LOG_WARN("Client not streaming after {} compute calls", compute_call_count);
    }
    return;
  }

  // Add this check for upstream readiness
  if (client_->isStreaming()) {
    static int last_upstream_log = 0;
    if (compute_call_count - last_upstream_log >= 60) {  // Log every 2 seconds at 30fps
      HOLOSCAN_LOG_INFO("Client streaming: YES, Upstream ready: {}",
                       client_->isUpstreamReady() ? "YES" : "NO");
      last_upstream_log = compute_call_count;
    }
  }

  // Check if frame sending is enabled
  if (!send_frames_.get()) {
    HOLOSCAN_LOG_DEBUG("Frame sending is disabled (send_frames=false)");
    return;
  }

  // Continue with frame processing since we know client is streaming

  // Add a small delay after connection to ensure upstream is ready
  static bool first_frame_after_connection = true;
  if (first_frame_after_connection && client_->isStreaming()) {
    HOLOSCAN_LOG_INFO("First frame after connection, waiting for upstream...");
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    first_frame_after_connection = false;
  }

  auto maybe_message = op_input.receive<holoscan::TensorMap>("input_frames");
  if (!maybe_message) {
    HOLOSCAN_LOG_DEBUG("No input message received");
    return;
  }

  auto message = maybe_message.value();

  // REFACTORED: Process all tensors in the message, but return early if any validation fails
  // This ensures the entire message is consumed from the queue
  bool message_processed_successfully = false;

  // Log tensor information for debugging
  for (const auto& [key, tensor] : message) {
    HOLOSCAN_LOG_INFO("=== TENSOR ANALYSIS ===");
    HOLOSCAN_LOG_INFO("Tensor key: {}", key);
    HOLOSCAN_LOG_INFO("Processing tensor: ndim={}, size={}, nbytes={}, device={}",
                     tensor->ndim(), tensor->size(), tensor->nbytes(),
                     (tensor->device().device_type == kDLCUDA ? "GPU" : "CPU"));

    // Get tensor shape
    auto shape = tensor->shape();
    std::string shape_str = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      shape_str += std::to_string(shape[i]);
      if (i < shape.size() - 1) shape_str += ", ";
    }
    shape_str += "]";
    HOLOSCAN_LOG_INFO("Tensor shape={}", shape_str);

    // 🔧 ENHANCED DEBUG: Write tensor to disk before processing - standardized to every 10th frame
    debug_frame_counter++;
    if (debug_frame_counter % 10 == 0) { // Save every 10th frame to match server frequency
      HOLOSCAN_LOG_INFO("💾 DEBUG: Writing input tensor to disk (frame {})", debug_frame_counter);
      writeTensorToDisk(tensor, "client_input_tensor", debug_frame_counter);
    }

    // FIXED: Replace continue with early return - validate tensor before processing
    if (!validateTensorData(tensor)) {
      HOLOSCAN_LOG_ERROR("Tensor validation failed, discarding entire message");
      return;  // Exit compute method entirely - message fully consumed
    }

    // FIXED: Replace continue with early return - process tensor based on data type
    if (tensor->dtype().code != kDLUInt || tensor->dtype().bits != 8) {
      HOLOSCAN_LOG_WARN("Unsupported tensor data type, discarding entire message");
      return;  // Exit compute method entirely - message fully consumed
    }

    // FIXED: Replace continue with early return - expect 3D tensor: [height, width, channels]
    if (tensor->ndim() != 3) {
      HOLOSCAN_LOG_WARN("Unexpected tensor dimensions: {}, expected 3D [height, width, channels], discarding entire message", tensor->ndim());
      return;  // Exit compute method entirely - message fully consumed
    }

    int tensor_height = static_cast<int>(shape[0]);
    int tensor_width = static_cast<int>(shape[1]);
    int channels = static_cast<int>(shape[2]);

    // CRITICAL: Validate tensor dimensions match client configuration
    int expected_width = static_cast<int>(width_.get());
    int expected_height = static_cast<int>(height_.get());

    HOLOSCAN_LOG_INFO("✅ Tensor validation passed: {}x{}x{}, {} bytes",
                     tensor_height, tensor_width, channels, tensor->nbytes());
    HOLOSCAN_LOG_INFO("Expected client dimensions: {}x{}", expected_width, expected_height);

    // FIXED: Replace continue with early return - check for dimension mismatch between tensor and client configuration
    if (tensor_width != expected_width || tensor_height != expected_height) {
      HOLOSCAN_LOG_ERROR("❌ DIMENSION MISMATCH: Tensor {}x{} does not match client configuration {}x{}",
                        tensor_width, tensor_height, expected_width, expected_height);
      HOLOSCAN_LOG_ERROR("   Holoscan Streaming Stack was negotiated for {}x{} but received {}x{}",
                        expected_width, expected_height, tensor_width, tensor_height);
      HOLOSCAN_LOG_ERROR("   This causes sendFrame() to fail silently in Holoscan Streaming Stack, discarding entire message");
      return;  // Exit compute method entirely - message fully consumed
    }

    // FIXED: Replace continue with early return - check channels
    if (channels != 3) {
      HOLOSCAN_LOG_WARN("Unexpected number of channels: {}, expected 3 for BGR, discarding entire message", channels);
      return;  // Exit compute method entirely - message fully consumed
    }

    HOLOSCAN_LOG_INFO("BGRA FRAME PROCESSING:");
    HOLOSCAN_LOG_INFO("  - Tensor data pointer: {}", tensor->data() ? "VALID" : "NULL");
    HOLOSCAN_LOG_INFO("  - Input BGR frame size: {} bytes", tensor->nbytes());
    HOLOSCAN_LOG_INFO("  - Tensor shape: {}", shape_str);
    HOLOSCAN_LOG_INFO("  - Converting BGR to BGRA format");
    HOLOSCAN_LOG_INFO("  - Using client configured dimensions: {}x{}", expected_width, expected_height);

    // IMPORTANT: Use configured client dimensions, not tensor dimensions
    size_t bgra_frame_size = expected_width * expected_height * 4;
    HOLOSCAN_LOG_INFO("  - Output BGRA frame size: {} bytes", bgra_frame_size);

    // 🔧 POTENTIAL FIX: Use shared_ptr to ensure data persistence
    // If VideoFrame doesn't copy data, we need to ensure the buffer stays alive
    auto bgra_buffer = std::make_shared<std::vector<uint8_t>>(bgra_frame_size);

    if (tensor->device().device_type == kDLCUDA) {
      // GPU tensor - copy to CPU first, then convert BGR to BGRA
      HOLOSCAN_LOG_INFO("Copying from GPU tensor to local buffer and converting to BGRA");

      std::vector<uint8_t> bgr_buffer(tensor->nbytes());

      // Copy from GPU to CPU
      CUDA_TRY(cudaMemcpy(bgr_buffer.data(),
                         tensor->data(),
                         tensor->nbytes(),
                         cudaMemcpyDeviceToHost));

      // Convert BGR to BGRA - use expected dimensions
      const uint8_t* bgr_data = bgr_buffer.data();
      for (int i = 0; i < expected_width * expected_height; ++i) {
        (*bgra_buffer)[i * 4 + 0] = bgr_data[i * 3 + 0];  // Blue
        (*bgra_buffer)[i * 4 + 1] = bgr_data[i * 3 + 1];  // Green
        (*bgra_buffer)[i * 4 + 2] = bgr_data[i * 3 + 2];  // Red
        (*bgra_buffer)[i * 4 + 3] = 255;                 // Alpha (opaque)
      }
    } else {
      // CPU tensor - direct conversion BGR to BGRA
      HOLOSCAN_LOG_INFO("Converting CPU BGR tensor to BGRA");

      const uint8_t* bgr_data = static_cast<const uint8_t*>(tensor->data());
      for (int i = 0; i < expected_width * expected_height; ++i) {
        (*bgra_buffer)[i * 4 + 0] = bgr_data[i * 3 + 0];  // Blue
        (*bgra_buffer)[i * 4 + 1] = bgr_data[i * 3 + 1];  // Green
        (*bgra_buffer)[i * 4 + 2] = bgr_data[i * 3 + 2];  // Red
        (*bgra_buffer)[i * 4 + 3] = 255;                 // Alpha (opaque)
      }
    }

    // Add content validation before sending
    size_t non_zero_count = 0;
    size_t check_bytes = std::min(bgra_buffer->size(), static_cast<size_t>(1000));
    for (size_t i = 0; i < check_bytes; ++i) {
      if ((*bgra_buffer)[i] != 0) non_zero_count++;
    }

    HOLOSCAN_LOG_INFO("Frame content analysis: {}/{} non-zero bytes in first {} bytes",
                     non_zero_count, check_bytes, check_bytes);

    // FIXED: Replace continue with early return - check minimum content
    if (non_zero_count < min_non_zero_bytes_.get()) {
      HOLOSCAN_LOG_WARN("Insufficient frame content: {}/{} non-zero bytes, discarding entire message",
                       non_zero_count, min_non_zero_bytes_.get());
      return;  // Exit compute method entirely - message fully consumed
    }

    HOLOSCAN_LOG_INFO("✅ Successfully converted BGR to BGRA: {} bytes", bgra_frame_size);
    HOLOSCAN_LOG_INFO("Frame data (first 5 BGRA pixels): {}, {}, {}, {}, {}",
                     (*bgra_buffer)[0], (*bgra_buffer)[1], (*bgra_buffer)[2], (*bgra_buffer)[3], (*bgra_buffer)[4]);

    // CRITICAL FIX: Create VideoFrame using configured client dimensions, not tensor dimensions
    uint64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // 🔧 POTENTIAL FIX: Ensure VideoFrame has its own copy of the data
    // Create VideoFrame - if it doesn't copy data internally, we need to ensure data persistence
    VideoFrame frame(static_cast<uint32_t>(expected_width),   // Use configured width (854)
                   static_cast<uint32_t>(expected_height),    // Use configured height (480)
                   bgra_buffer->data(),
                   bgra_frame_size,
                   timestamp);

    // Set format after construction
    frame.setFormat(PixelFormat::BGRA);

    // 🔧 CRITICAL DEBUG: Verify the VideoFrame actually contains the data we expect
    HOLOSCAN_LOG_INFO("🔍 POST-CONSTRUCTION VERIFICATION:");
    HOLOSCAN_LOG_INFO("  - VideoFrame width: {}", frame.getWidth());
    HOLOSCAN_LOG_INFO("  - VideoFrame height: {}", frame.getHeight());
    HOLOSCAN_LOG_INFO("  - VideoFrame data size: {}", frame.getDataSize());
    HOLOSCAN_LOG_INFO("  - VideoFrame format: {}", static_cast<int>(frame.getFormat()));
    HOLOSCAN_LOG_INFO("  - VideoFrame data pointer: {}", static_cast<const void*>(frame.getData()));
    HOLOSCAN_LOG_INFO("  - Original bgra_buffer pointer: {}", static_cast<const void*>(bgra_buffer->data()));

    // Verify the frame data is accessible and valid
    const uint8_t* frame_data = frame.getData();
    if (frame_data && frame.getDataSize() > 4) {
        HOLOSCAN_LOG_INFO("  - VideoFrame first 5 bytes: {}, {}, {}, {}, {}",
                         frame_data[0], frame_data[1], frame_data[2], frame_data[3], frame_data[4]);
    } else {
        HOLOSCAN_LOG_ERROR("  - VideoFrame data is not accessible!");
    }

    HOLOSCAN_LOG_INFO("✅ VideoFrame created with configured dimensions: {}x{}, format={}, size={} bytes",
                     frame.getWidth(), frame.getHeight(),
                     static_cast<int>(frame.getFormat()), frame.getDataSize());

    // 🔧 ENHANCED DEBUG: Write VideoFrame to disk before sending - standardized to every 10th frame
    if (debug_frame_counter % 10 == 0) { // Save every 10th frame to match server frequency
      HOLOSCAN_LOG_INFO("💾 DEBUG: Writing VideoFrame to disk before sending (frame {})", debug_frame_counter);

      // 🔍 ENHANCED DEBUGGING: Validate frame data before writing to disk
      const uint8_t* frame_data_ptr = frame.getData();
      size_t frame_data_size = frame.getDataSize();

      HOLOSCAN_LOG_INFO("📊 PRE-WRITE DEBUG:");
      HOLOSCAN_LOG_INFO("  - bgra_buffer.data(): {}", static_cast<const void*>(bgra_buffer->data()));
      HOLOSCAN_LOG_INFO("  - frame.getData(): {}", static_cast<const void*>(frame_data_ptr));
      HOLOSCAN_LOG_INFO("  - Same pointer: {}", (frame_data_ptr == bgra_buffer->data()) ? "YES" : "NO");
      HOLOSCAN_LOG_INFO("  - bgra_buffer size: {}", bgra_buffer->size());
      HOLOSCAN_LOG_INFO("  - frame data size: {}", frame_data_size);
      HOLOSCAN_LOG_INFO("  - bgra_buffer first 5 bytes: {}, {}, {}, {}, {}",
                       (*bgra_buffer)[0], (*bgra_buffer)[1], (*bgra_buffer)[2], (*bgra_buffer)[3], (*bgra_buffer)[4]);

      if (frame_data_ptr && frame_data_size > 4) {
        HOLOSCAN_LOG_INFO("  - frame data first 5 bytes: {}, {}, {}, {}, {}",
                         frame_data_ptr[0], frame_data_ptr[1], frame_data_ptr[2], frame_data_ptr[3], frame_data_ptr[4]);
      } else {
        HOLOSCAN_LOG_ERROR("  - frame data is NULL or too small!");
      }

      // Write both the buffer directly and the frame for comparison
      writeFrameToDisk(frame, "client_output_videoframe", debug_frame_counter);

      // 🔧 ADDITIONAL DEBUG: Write bgra_buffer directly to compare
      try {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;

        std::stringstream filename;
        filename << "client_bgra_buffer_direct_frame"
                 << std::setfill('0') << std::setw(6) << debug_frame_counter
                 << "_" << std::put_time(std::localtime(&time_t), "%H%M%S")
                 << "_" << std::setfill('0') << std::setw(3) << ms.count();

        // Write raw buffer data directly
        std::string raw_filename = filename.str() + "_direct_raw.bin";
        std::ofstream raw_file(raw_filename, std::ios::binary);
        if (raw_file.is_open()) {
          raw_file.write(reinterpret_cast<const char*>(bgra_buffer->data()), bgra_buffer->size());
          raw_file.close();
          HOLOSCAN_LOG_INFO("📁 Wrote direct bgra_buffer to: {}", raw_filename);
        }

        // Write metadata
        std::string meta_filename = filename.str() + "_direct_meta.txt";
        std::ofstream meta_file(meta_filename);
        if (meta_file.is_open()) {
          meta_file << "Direct BGRA Buffer Metadata:\n";
          meta_file << "Width: " << expected_width << "\n";
          meta_file << "Height: " << expected_height << "\n";
          meta_file << "Buffer Size: " << bgra_buffer->size() << " bytes\n";
          meta_file << "Format: BGRA\n";
          meta_file << "First 10 pixels (direct from buffer):\n";
          for (int i = 0; i < 10 && (i * 4 + 3) < bgra_buffer->size(); ++i) {
            meta_file << "Pixel " << i << ": "
                     << static_cast<int>((*bgra_buffer)[i * 4 + 0]) << " "
                     << static_cast<int>((*bgra_buffer)[i * 4 + 1]) << " "
                     << static_cast<int>((*bgra_buffer)[i * 4 + 2]) << " "
                     << static_cast<int>((*bgra_buffer)[i * 4 + 3]) << "\n";
          }
          meta_file.close();
          HOLOSCAN_LOG_INFO("📁 Wrote direct buffer metadata to: {}", meta_filename);
        }
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Error writing direct buffer: {}", e.what());
      }
    }

    // FIXED: Replace continue with early return - validate frame before sending
    if (frame.getWidth() == 0 || frame.getHeight() == 0) {
      HOLOSCAN_LOG_ERROR("Frame has invalid dimensions after creation, discarding entire message");
      return;  // Exit compute method entirely - message fully consumed
    }

    if (!frame.isValid() || frame.getDataSize() == 0) {
      HOLOSCAN_LOG_ERROR("Frame validation failed after construction, discarding entire message");
      return;  // Exit compute method entirely - message fully consumed
    }

    // CRITICAL: Add comprehensive pre-send frame validation
    bool frame_validation_passed = true;
    std::string validation_errors;

    // Check frame dimensions
    if (frame.getWidth() == 0 || frame.getHeight() == 0) {
      validation_errors += "Invalid dimensions; ";
      frame_validation_passed = false;
    }

    // Check frame data pointer and size
    const uint8_t* frame_data_ptr = frame.getData();
    if (!frame_data_ptr) {
      validation_errors += "Null data pointer; ";
      frame_validation_passed = false;
    }

    if (frame.getDataSize() == 0) {
      validation_errors += "Zero data size; ";
      frame_validation_passed = false;
    }

    // Check expected vs actual frame size
    size_t expected_size = frame.getWidth() * frame.getHeight() * 4;  // BGRA = 4 bytes per pixel
    if (frame.getDataSize() != expected_size) {
      validation_errors += fmt::format("Size mismatch (expected {}, got {}); ", expected_size, frame.getDataSize());
      frame_validation_passed = false;
    }

    // Check frame format
    if (frame.getFormat() != PixelFormat::BGRA) {
      validation_errors += fmt::format("Wrong format (expected BGRA, got {}); ", static_cast<int>(frame.getFormat()));
      frame_validation_passed = false;
    }

    // Check if frame has actual content (not all zeros)
    if (frame_data_ptr && frame.getDataSize() > 0) {
      size_t non_zero_bytes = 0;
      size_t check_bytes = std::min(frame.getDataSize(), static_cast<size_t>(1000));
      for (size_t i = 0; i < check_bytes; ++i) {
        if (frame_data_ptr[i] != 0) non_zero_bytes++;
      }

      if (non_zero_bytes < min_non_zero_bytes_.get()) {
        validation_errors += fmt::format("Insufficient content ({}/{} non-zero bytes); ", non_zero_bytes, min_non_zero_bytes_.get());
        frame_validation_passed = false;
      }

      HOLOSCAN_LOG_INFO("Frame content validation: {}/{} non-zero bytes in first {} bytes",
                       non_zero_bytes, check_bytes, check_bytes);
    }

    // FIXED: Replace continue with early return - check frame validation
    if (!frame_validation_passed) {
      HOLOSCAN_LOG_ERROR("❌ FRAME VALIDATION FAILED: {}", validation_errors);
      HOLOSCAN_LOG_ERROR("Frame details: {}x{}, format={}, size={}, data_ptr={}",
                        frame.getWidth(), frame.getHeight(),
                        static_cast<int>(frame.getFormat()), frame.getDataSize(),
                        frame.getData() ? "VALID" : "NULL");
      HOLOSCAN_LOG_ERROR("Discarding entire message due to frame validation failure");
      return;  // Exit compute method entirely - message fully consumed
    }

    HOLOSCAN_LOG_INFO("✅ Frame validation passed - ready for transmission");

    // 🔧 NEW: Output BGRA tensor for HoloViz BEFORE network transmission
    try {
      // Create a GXF entity for BGRA tensor output
      auto maybe_bgra_entity = nvidia::gxf::Entity::New(context.context());
      if (maybe_bgra_entity) {
        // Add a tensor to the entity
        auto bgra_tensor_handle = maybe_bgra_entity.value().add<nvidia::gxf::Tensor>("bgra_tensor");
        if (bgra_tensor_handle) {
          // Set up dimensions and shape for BGRA format [Height, Width, Channels]
          nvidia::gxf::Shape bgra_shape = nvidia::gxf::Shape{
            static_cast<int32_t>(expected_height),  // 480
            static_cast<int32_t>(expected_width),   // 854
            4  // BGRA has 4 channels
          };

          // Set up element type
          nvidia::gxf::PrimitiveType element_type = nvidia::gxf::PrimitiveType::kUnsigned8;
          int element_size = nvidia::gxf::PrimitiveTypeSize(element_type);

          // Create a copy of the BGRA data in host memory for the tensor
          auto bgra_tensor_data = std::shared_ptr<uint8_t[]>(new uint8_t[bgra_frame_size]);
          if (bgra_tensor_data) {
            // Copy the BGRA data
            std::memcpy(bgra_tensor_data.get(), bgra_buffer->data(), bgra_frame_size);

            // Wrap the memory in the tensor
            bgra_tensor_handle.value()->wrapMemory(
                bgra_shape,
                element_type,
                element_size,
                nvidia::gxf::ComputeTrivialStrides(bgra_shape, element_size),
                nvidia::gxf::MemoryStorageType::kSystem,
                bgra_tensor_data.get(),
                [bgra_tensor_data](void*) mutable {
                  // The shared_ptr will automatically clean up when it goes out of scope
                  bgra_tensor_data.reset();
                  return nvidia::gxf::Success;
                });

            // Emit the BGRA tensor entity to HoloViz
            op_output.emit(maybe_bgra_entity.value(), "output_frames");
            HOLOSCAN_LOG_INFO("✅ Emitted BGRA tensor for HoloViz: [{}x{}x4], {} bytes",
                             expected_height, expected_width, bgra_frame_size);
          } else {
            HOLOSCAN_LOG_ERROR("Failed to allocate memory for BGRA tensor");
          }
        } else {
          HOLOSCAN_LOG_ERROR("Failed to add BGRA tensor to entity");
        }
      } else {
        HOLOSCAN_LOG_ERROR("Failed to create BGRA entity for HoloViz");
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Exception creating BGRA tensor output: {}", e.what());
    }

    // Add retry logic for frame sending
    bool send_success = false;
    int send_attempts = 0;
    const int max_send_attempts = 3;

    while (!send_success && send_attempts < max_send_attempts) {
      try {
        // Check if client is ready to send frames (both streaming and upstream ready)
        if (client_ && client_->isStreaming() && client_->isUpstreamReady()) {
          HOLOSCAN_LOG_INFO("Attempting to send frame: {}x{}, {} bytes, streaming=active, upstream=ready",
                           frame.getWidth(), frame.getHeight(), frame.getDataSize());

          // Add detailed pre-send validation
          HOLOSCAN_LOG_DEBUG("Pre-send validation:");
          HOLOSCAN_LOG_DEBUG("  - Frame valid: {}", frame.isValid() ? "YES" : "NO");
          HOLOSCAN_LOG_DEBUG("  - Frame data: {}", frame.getData() ? "VALID" : "NULL");
          HOLOSCAN_LOG_DEBUG("  - Frame size: {} bytes", frame.getDataSize());
          HOLOSCAN_LOG_DEBUG("  - Client streaming: {}", client_->isStreaming() ? "YES" : "NO");
          HOLOSCAN_LOG_DEBUG("  - Upstream ready: {}", client_->isUpstreamReady() ? "YES" : "NO");

          send_success = client_->sendFrame(frame);
          if (send_success) {
            HOLOSCAN_LOG_INFO("✅ Frame sent successfully on attempt {}",
                             send_attempts + 1);
            // Reset retry counter on successful send
            retry_count_ = 0;
            message_processed_successfully = true;
          } else {
            send_attempts++;

            // CRITICAL: Add detailed error investigation
            HOLOSCAN_LOG_ERROR("❌ Frame send failed, attempt {}/{} - INVESTIGATING CAUSE:",
                             send_attempts, max_send_attempts);
            HOLOSCAN_LOG_ERROR("  - Frame properties: {}x{}, format={}, {} bytes",
                              frame.getWidth(), frame.getHeight(),
                              static_cast<int>(frame.getFormat()), frame.getDataSize());
            HOLOSCAN_LOG_ERROR("  - Frame validation: valid={}, data_ptr={}",
                              frame.isValid() ? "YES" : "NO",
                              frame.getData() ? "VALID" : "NULL");
            HOLOSCAN_LOG_ERROR("  - Client state: streaming={}, upstream_ready={}",
                              client_->isStreaming() ? "YES" : "NO",
                              client_->isUpstreamReady() ? "YES" : "NO");

            // Check if it's a connection issue
            if (!client_->isStreaming()) {
              HOLOSCAN_LOG_ERROR("  - ROOT CAUSE: Client lost streaming connection");
              break;  // Exit retry loop for connection issues
            } else if (!client_->isUpstreamReady()) {
              HOLOSCAN_LOG_ERROR("  - ROOT CAUSE: Upstream connection not ready");
              HOLOSCAN_LOG_ERROR("  - Client connected but upstream channel not established");
            } else {
              HOLOSCAN_LOG_ERROR("  - POSSIBLE CAUSE: Client reports streaming and upstream ready but sendFrame fails");
              HOLOSCAN_LOG_ERROR("  - This could be Holoscan Streaming Stack internal error or frame format issue");
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(10));
          }
        } else if (client_ && client_->isStreaming() && !client_->isUpstreamReady()) {
          HOLOSCAN_LOG_WARN("Cannot send frame: client is streaming but upstream not ready");
          HOLOSCAN_LOG_WARN("  - Client streaming: YES");
          HOLOSCAN_LOG_WARN("  - Upstream ready: NO");
          send_attempts++;
          std::this_thread::sleep_for(std::chrono::milliseconds(100));  // Brief wait
        } else {
          HOLOSCAN_LOG_ERROR("Cannot send frame: client not ready");
          HOLOSCAN_LOG_ERROR("  - Client exists: {}", client_ ? "YES" : "NO");
          if (client_) {
            HOLOSCAN_LOG_ERROR("  - Client streaming: {}", client_->isStreaming() ? "YES" : "NO");
            HOLOSCAN_LOG_ERROR("  - Upstream ready: {}", client_->isUpstreamReady() ? "YES" : "NO");
          }
          break;  // Exit retry loop if client is fundamentally not ready
        }
      } catch (const std::exception& e) {
        send_attempts++;
        HOLOSCAN_LOG_ERROR("❌ Exception during frame send attempt {}: {}",
                          send_attempts, e.what());
        if (send_attempts < max_send_attempts) {
          std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
      }
    }

    if (!send_success) {
      HOLOSCAN_LOG_ERROR("❌ Failed to send frame after {} attempts", max_send_attempts);
      // Don't return here - we've still consumed the message successfully
    }

    // Small delay to control frame rate
    std::this_thread::sleep_for(std::chrono::microseconds(100));

    // Since we're processing only the first valid tensor, break after processing
    break;
  }

  // Output received frames if we have any
  if (receive_frames_.get()) {
    std::lock_guard<std::mutex> lock(frame_mutex_);
    if (has_new_frame_) {
      try {
        auto& frame_ref = current_frame_;  // Reference to avoid copying

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

  // 🔧 FIX: Add debugging for client frame reception
  HOLOSCAN_LOG_INFO("🎯 CLIENT: Frame received callback triggered! Frame: {}x{}, {} bytes",
                   frame.getWidth(), frame.getHeight(), frame.getDataSize());

  // Store the new frame
  current_frame_ = frame;
  has_new_frame_ = true;

  // ENHANCED DEBUG: Write received frames from server to disk for visualization
  static int received_frame_counter = 0;
  received_frame_counter++;

  HOLOSCAN_LOG_INFO("📥 CLIENT: Received frame #{} from server: {}x{}",
                   received_frame_counter, frame.getWidth(), frame.getHeight());

  // Write every 10th frame to disk to match server frequency
  if (received_frame_counter % 10 == 0) {
    HOLOSCAN_LOG_INFO("💾 DEBUG: Writing frame received from server to disk (frame {})", received_frame_counter);

    try {
      // Generate filename with timestamp and frame number
      auto now = std::chrono::system_clock::now();
      auto time_t = std::chrono::system_clock::to_time_t(now);
      auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
          now.time_since_epoch()) % 1000;

      std::stringstream filename;
      filename << "debug_client_received_from_server_frame"
               << std::setfill('0') << std::setw(6) << received_frame_counter
               << "_" << std::put_time(std::localtime(&time_t), "%H%M%S")
               << "_" << std::setfill('0') << std::setw(3) << ms.count();

      // Get frame properties
      uint32_t width = frame.getWidth();
      uint32_t height = frame.getHeight();
      const uint8_t* data = frame.getData();
      size_t data_size = frame.getDataSize();
      auto format = frame.getFormat();

      if (data && data_size > 0 && width > 0 && height > 0) {
        // Write raw binary data
        std::string raw_filename = filename.str() + "_raw.bin";
        std::ofstream raw_file(raw_filename, std::ios::binary);
        if (raw_file.is_open()) {
          raw_file.write(reinterpret_cast<const char*>(data), data_size);
          raw_file.close();
          HOLOSCAN_LOG_INFO("📁 Wrote raw frame data received from server to: {}", raw_filename);
        }

        // Write metadata file
        std::string meta_filename = filename.str() + "_meta.txt";
        std::ofstream meta_file(meta_filename);
        if (meta_file.is_open()) {
          meta_file << "Frame Received from Server Metadata:\n";
          meta_file << "Width: " << width << "\n";
          meta_file << "Height: " << height << "\n";
          meta_file << "Data Size: " << data_size << " bytes\n";
          meta_file << "Pixel Format: " << static_cast<int>(format) << " (";
          switch (format) {
            case PixelFormat::BGR: meta_file << "BGR"; break;
            case PixelFormat::BGRA: meta_file << "BGRA"; break;
            case PixelFormat::RGBA: meta_file << "RGBA"; break;
            default: meta_file << "Unknown"; break;
          }
          meta_file << ")\n";
          meta_file << "Timestamp: " << frame.getTimestamp() << "\n";
          meta_file << "Bytes per pixel: " << (data_size / (width * height)) << "\n";

          // Add first few pixel values for inspection
          meta_file << "\nFirst 10 pixels received from server (raw bytes):\n";
          size_t bytes_per_pixel = (format == PixelFormat::BGRA || format == PixelFormat::RGBA) ? 4 : 3;
          for (int i = 0; i < std::min(10, static_cast<int>(width * height)) && i * bytes_per_pixel < data_size; ++i) {
            meta_file << "Pixel " << i << ": ";
            for (size_t j = 0; j < bytes_per_pixel && (i * bytes_per_pixel + j) < data_size; ++j) {
              meta_file << static_cast<int>(data[i * bytes_per_pixel + j]) << " ";
            }
            meta_file << "\n";
          }

          meta_file.close();
          HOLOSCAN_LOG_INFO("📁 Wrote frame metadata received from server to: {}", meta_filename);
        }

        // Write as PPM image file (for easy viewing)
        if (format == PixelFormat::BGRA || format == PixelFormat::BGR || format == PixelFormat::RGBA) {
          std::string ppm_filename = filename.str() + ".ppm";
          std::ofstream ppm_file(ppm_filename, std::ios::binary);
          if (ppm_file.is_open()) {
            // PPM header
            ppm_file << "P6\n" << width << " " << height << "\n255\n";

            // Convert pixel data to RGB for PPM
            size_t bytes_per_pixel = (format == PixelFormat::BGRA || format == PixelFormat::RGBA) ? 4 : 3;
            for (uint32_t y = 0; y < height; ++y) {
              for (uint32_t x = 0; x < width; ++x) {
                size_t pixel_offset = (y * width + x) * bytes_per_pixel;
                if (pixel_offset + 2 < data_size) {
                  uint8_t r, g, b;
                  if (format == PixelFormat::BGRA || format == PixelFormat::BGR) {
                    // BGR(A) format - swap B and R
                    b = data[pixel_offset + 0];
                    g = data[pixel_offset + 1];
                    r = data[pixel_offset + 2];
                  } else {
                    // RGB(A) format
                    r = data[pixel_offset + 0];
                    g = data[pixel_offset + 1];
                    b = data[pixel_offset + 2];
                  }
                  ppm_file.write(reinterpret_cast<const char*>(&r), 1);
                  ppm_file.write(reinterpret_cast<const char*>(&g), 1);
                  ppm_file.write(reinterpret_cast<const char*>(&b), 1);
                }
              }
            }
            ppm_file.close();
            HOLOSCAN_LOG_INFO("🖼️ Wrote PPM image received from server to: {} (can be opened with image viewers)", ppm_filename);
          }
        }
      } else {
        HOLOSCAN_LOG_ERROR("Invalid frame data received from server");
      }
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_ERROR("Exception writing received frame to disk: {}", e.what());
    }
  }
}

// Remove the generateFrame() method

// Adjust the initialize method to not set a frame source
// Remove the line: client_->setFrameSource([this]() { return generateFrame(); });

// Ensure the compute method processes tensors correctly
// No changes needed in compute as it already processes tensors from the previous operator.

bool StreamingClientOp::validateTensorData(const std::shared_ptr<holoscan::Tensor>& tensor) {
  if (!tensor) {
    HOLOSCAN_LOG_ERROR("Tensor validation failed: null tensor");
    return false;
  }

  // Check tensor dimensions
  if (tensor->ndim() != 3) {
    HOLOSCAN_LOG_ERROR("Tensor validation failed: expected 3D tensor, got {} dimensions", tensor->ndim());
    return false;
  }

  auto shape = tensor->shape();
  if (shape.size() != 3) {
    HOLOSCAN_LOG_ERROR("Tensor validation failed: invalid shape size {}", shape.size());
    return false;
  }

  int height = static_cast<int>(shape[0]);
  int width = static_cast<int>(shape[1]);
  int channels = static_cast<int>(shape[2]);

  // Validate dimensions
  if (width <= 0 || height <= 0 || channels != 3) {
    HOLOSCAN_LOG_ERROR("Tensor validation failed: invalid dimensions {}x{}x{}", height, width, channels);
    return false;
  }

  // Check data type
  if (tensor->dtype().code != kDLUInt || tensor->dtype().bits != 8) {
    HOLOSCAN_LOG_ERROR("Tensor validation failed: expected uint8, got dtype code={}, bits={}",
                       static_cast<int>(tensor->dtype().code), tensor->dtype().bits);
    return false;
  }

  // Check data pointer
  if (!tensor->data()) {
    HOLOSCAN_LOG_ERROR("Tensor validation failed: null data pointer");
    return false;
  }

  // Check data size
  size_t expected_size = width * height * channels;
  if (tensor->nbytes() != expected_size) {
    HOLOSCAN_LOG_ERROR("Tensor validation failed: size mismatch, expected {} bytes, got {}",
                       expected_size, tensor->nbytes());
    return false;
  }

  // Check for valid content (not all zeros)
  const uint8_t* data_ptr = static_cast<const uint8_t*>(tensor->data());
  bool has_valid_data = false;

  if (tensor->device().device_type == kDLCUDA) {
    // For GPU tensors, copy a small sample to check
    std::vector<uint8_t> sample_data(1000);
    size_t check_size = std::min(static_cast<size_t>(tensor->nbytes()), static_cast<size_t>(1000));

    cudaError_t cuda_result = cudaMemcpy(sample_data.data(), data_ptr, check_size, cudaMemcpyDeviceToHost);
    if (cuda_result == cudaSuccess) {
      for (size_t i = 0; i < check_size; ++i) {
        if (sample_data[i] != 0) {
          has_valid_data = true;
          break;
        }
      }
    } else {
      HOLOSCAN_LOG_ERROR("Tensor validation failed: CUDA memcpy error: {}", cudaGetErrorString(cuda_result));
      return false;
    }
  } else {
    // For CPU tensors, check directly
    size_t check_size = std::min(static_cast<size_t>(tensor->nbytes()), static_cast<size_t>(1000));
    for (size_t i = 0; i < check_size; ++i) {
      if (data_ptr[i] != 0) {
        has_valid_data = true;
        break;
      }
    }
  }

  if (!has_valid_data) {
    HOLOSCAN_LOG_WARN("Tensor validation warning: first 1000 bytes are all zeros (may indicate empty frame)");
    // Don't fail validation, but warn - empty frames during startup are common
  }

  HOLOSCAN_LOG_DEBUG("✅ Tensor validation passed: {}x{}x{}, {} bytes, device={}",
                     height, width, channels, tensor->nbytes(),
                     tensor->device().device_type == kDLCUDA ? "GPU" : "CPU");

  return true;
}

}  // namespace holoscan::ops






