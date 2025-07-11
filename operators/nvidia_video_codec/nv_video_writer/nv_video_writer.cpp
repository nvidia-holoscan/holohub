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

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/gxf/entity.hpp"

#include "gxf/core/entity.hpp"    // nvidia::gxf::Entity::Shared
#include "gxf/std/allocator.hpp"  // nvidia::gxf::Allocator, nvidia::gxf::MemoryStorageType
#include "gxf/std/tensor.hpp"     // nvidia::gxf::Tensor etc.
#include "gxf/std/timestamp.hpp"  // nvidia::gxf::Timestamp

namespace holoscan::ops {

void NvVideoWriterOp::setup(OperatorSpec& spec) {
  spec.input<holoscan::gxf::Entity>("input");

  spec.param(output_file_,
             "output_file",
             "Output File",
             "Output file to write video frames to",
             std::string(""));
  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(verbose_, "verbose", "Verbose", "Print detailed writer information", false);
}

void NvVideoWriterOp::initialize() {
  Operator::initialize();

  // Validate output file path
  if (output_file_.get().empty()) {
    throw std::runtime_error("Output file path cannot be empty");
  }

  // Create directory if it doesn't exist
  std::filesystem::path file_path(output_file_.get());
  if (file_path.has_parent_path()) {
    std::filesystem::create_directories(file_path.parent_path());
  }

  // Open output file for binary writing
  output_stream_.open(output_file_.get(), std::ios::out | std::ios::binary);
  if (!output_stream_) {
    throw std::runtime_error("Failed to open output file: " + output_file_.get());
  }

  // Reset statistics
  frame_count_ = 0;
  total_bytes_written_ = 0;

  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("Video writer initialized. Output file: {}", output_file_.get());
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

  // Get tensor data and size
  const uint8_t* data = static_cast<const uint8_t*>(tensor->data());
  size_t data_size = tensor->size();

  if (!data || data_size == 0) {
    HOLOSCAN_LOG_WARN("Received empty frame data, skipping");
    return;
  }

  // Write encoded frame data directly to file
  try {
    output_stream_.write(reinterpret_cast<const char*>(data), data_size);
    output_stream_.flush();  // Ensure data is written immediately

    frame_count_++;
    total_bytes_written_ += data_size;

    if (verbose_.get()) {
      HOLOSCAN_LOG_INFO("Wrote frame {} ({} bytes, total: {} bytes)",
                        frame_count_,
                        data_size,
                        total_bytes_written_);
    }
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Failed to write frame data: {}", e.what());
    throw std::runtime_error("Failed to write encoded frame to file");
  }
}

void NvVideoWriterOp::stop() {
  // Close output file
  if (output_stream_.is_open()) {
    output_stream_.close();
  }

  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("Video writer stopped. Total frames: {}, Total bytes: {}",
                      frame_count_,
                      total_bytes_written_);
  }
}

}  // namespace holoscan::ops
