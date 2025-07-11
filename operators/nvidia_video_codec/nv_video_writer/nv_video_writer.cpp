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

#include "nv_video_writer.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/gxf/entity.hpp"

#include "gxf/core/entity.hpp"    // nvidia::gxf::Entity::Shared
#include "gxf/std/allocator.hpp"  // nvidia::gxf::Allocator, nvidia::gxf::MemoryStorageType
#include "gxf/std/tensor.hpp"     // nvidia::gxf::Tensor etc.
#include "gxf/std/timestamp.hpp"  // nvidia::gxf::Timestamp

namespace holoscan::ops {

NvVideoWriterOp::~NvVideoWriterOp() {
  // Ensure file is closed properly
  if (output_stream_.is_open()) {
    output_stream_.close();
  }
}

void NvVideoWriterOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");

  spec.param(output_file_,
             "output_file",
             "Output File",
             "Output file to write video frames to",
             std::string(""));
  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(verbose_, "verbose", "Verbose", "Print detailed writer information", false);
  spec.param(buffer_size_,
             "buffer_size",
             "Buffer Size",
             "Buffer size for file I/O operations (bytes)",
             DEFAULT_BUFFER_SIZE);
}

void NvVideoWriterOp::initialize() {
  Operator::initialize();

  // Validate output file path
  if (output_file_.get().empty()) {
    throw std::runtime_error("Output file path cannot be empty");
  }

  validateOutputFile(output_file_.get());

  // Create directory if it doesn't exist
  std::filesystem::path file_path(output_file_.get());
  if (file_path.has_parent_path()) {
    std::filesystem::create_directories(file_path.parent_path());
  }

  // Open output file for binary writing with custom buffer size
  output_stream_.open(output_file_.get(), std::ios::out | std::ios::binary);
  if (!output_stream_) {
    throw std::runtime_error("Failed to open output file: " + output_file_.get());
  }

  // Set custom buffer size for better performance
  if (buffer_size_.get() > 0) {
    output_stream_.rdbuf()->pubsetbuf(nullptr, static_cast<std::streamsize>(buffer_size_.get()));
  }

  // Reset statistics
  frame_count_ = 0;
  total_bytes_written_ = 0;
  start_time_ = std::chrono::steady_clock::now();
  last_log_time_ = start_time_;
  frames_since_last_log_ = 0;

  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("Video writer initialized. Output file: {}, Buffer size: {} bytes",
                      output_file_.get(),
                      buffer_size_.get());
  }
}

void NvVideoWriterOp::compute(InputContext& op_input, OutputContext& op_output,
                              ExecutionContext& context) {
  // Get input entity containing encoded frame data
  auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
  if (!maybe_entity) {
    throw std::runtime_error("Failed to receive input entity");
  }

  // Get the tensor from the input message
  auto tensor = maybe_entity.value().get<Tensor>("");
  if (!tensor) {
    throw std::runtime_error("Failed to get tensor from input message");
  }

  if (tensor->dtype().code != DLDataTypeCode::kDLUInt) {
    HOLOSCAN_LOG_WARN("Expected uint8 tensor data, got type: {}",
                      static_cast<int>(tensor->dtype().code));
  }

  // Get tensor data and size
  const uint8_t* data = static_cast<const uint8_t*>(tensor->data());
  size_t data_size = tensor->size();

  if (!data || data_size == 0) {
    HOLOSCAN_LOG_WARN("Received empty frame data, skipping");
    return;
  }

  // Validate reasonable frame size (basic sanity check)
  if (data_size > 100 * 1024 * 1024) {  // 100MB seems too large for a single frame
    HOLOSCAN_LOG_WARN("Frame size unusually large: {} bytes", data_size);
  }

  // Write encoded frame data directly to file
  try {
    output_stream_.write(reinterpret_cast<const char*>(data), data_size);

    // Check if write was successful
    if (!output_stream_.good()) {
      throw std::runtime_error("Failed to write frame data to file");
    }

    frame_count_++;
    total_bytes_written_ += data_size;
    frames_since_last_log_++;

    // Throttled verbose logging for performance
    if (verbose_.get() && frames_since_last_log_ >= LOG_INTERVAL_FRAMES) {
      logPerformanceStats();
      frames_since_last_log_ = 0;
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to write frame data: {}", e.what());
    throw std::runtime_error("Failed to write encoded frame to file");
  }
}

void NvVideoWriterOp::stop() {
  if (output_stream_.is_open()) {
    try {
      // Explicitly flush any remaining buffered data to ensure data integrity
      output_stream_.flush();
    } catch (const std::exception& e) {
      HOLOSCAN_LOG_WARN("Error flushing video file during stop: {}", e.what());
    }
    output_stream_.close();
  }

  if (verbose_.get()) {
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time_);
    double fps = frame_count_ > 0 ? (frame_count_ * 1000.0) / duration.count() : 0.0;

    HOLOSCAN_LOG_INFO(
        "Video writer stopped. Total frames: {}, Total bytes: {}, "
        "Duration: {}ms, Average FPS: {:.2f}",
        frame_count_,
        total_bytes_written_,
        duration.count(),
        fps);
  }
}

void NvVideoWriterOp::validateOutputFile(const std::string& filepath) {
  // Check if file path is reasonable
  if (filepath.length() > 4096) {
    throw std::runtime_error("Output file path too long");
  }

  // Validate file extension for video formats
  std::string lower_path = filepath;
  std::transform(lower_path.begin(), lower_path.end(), lower_path.begin(), ::tolower);

  // Check for recognized video file extensions (C++17 compatible)
  const std::vector<std::string> valid_extensions = {
      ".h264", ".h265", ".264", ".265", ".hevc", ".mp4"};

  bool valid_extension = false;
  for (const auto& ext : valid_extensions) {
    if (lower_path.length() >= ext.length() &&
        lower_path.substr(lower_path.length() - ext.length()) == ext) {
      valid_extension = true;
      break;
    }
  }

  if (!valid_extension) {
    HOLOSCAN_LOG_WARN(
        "Output file '{}' doesn't have a recognized video extension. "
        "Recommended: .h264, .h265, .264, .265, .hevc, .mp4",
        filepath);
  }

  // Check if parent directory is writable
  std::filesystem::path file_path(filepath);
  if (file_path.has_parent_path()) {
    auto parent_path = file_path.parent_path();
    if (std::filesystem::exists(parent_path) && !std::filesystem::is_directory(parent_path)) {
      throw std::runtime_error("Parent path exists but is not a directory: " +
                               parent_path.string());
    }
  }
}

void NvVideoWriterOp::logPerformanceStats() {
  auto current_time = std::chrono::steady_clock::now();
  auto duration =
      std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_log_time_);

  double fps = duration.count() > 0 ? (LOG_INTERVAL_FRAMES * 1000.0) / duration.count() : 0.0;

  HOLOSCAN_LOG_INFO("Wrote {} frames (avg FPS: {:.2f}, total: {} frames, {} bytes)",
                    LOG_INTERVAL_FRAMES,
                    fps,
                    frame_count_,
                    total_bytes_written_);

  last_log_time_ = current_time;
}

}  // namespace holoscan::ops
