/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "nv_video_reader.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "holoscan/core/execution_context.hpp"
#include "holoscan/core/executor.hpp"
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/io_context.hpp"

#include "gxf/std/tensor.hpp"     // nvidia::gxf::Tensor etc.
#include "gxf/std/timestamp.hpp"  // nvidia::gxf::Timestamp

namespace holoscan::ops {

void NvVideoReaderOp::setup(OperatorSpec& spec) {
  spec.output<holoscan::gxf::Entity>("output");

  spec.param(directory_,
             "directory",
             "Directory",
             "Directory containing the video files to read",
             std::string(""));
  spec.param(
      filename_, "filename", "Filename", "Filename of the video file to read", std::string(""));
  spec.param(allocator_, "allocator", "Allocator", "Allocator for output buffers.");
  spec.param(loop_, "loop", "Loop", "Loop the video file when end is reached", false);
  spec.param(verbose_, "verbose", "Verbose", "Print detailed reader information", false);
}

void NvVideoReaderOp::initialize() {
  Operator::initialize();

  filepath_ = (std::filesystem::path(directory_.get()) / filename_.get()).string();

  // Check if file exists
  std::ifstream file(filepath_);
  if (!file.good()) {
    throw std::runtime_error("Cannot open video file: " + filepath_);
  }
  file.close();

  // Initialize FFmpeg demuxer
  try {
    demuxer_ = std::make_unique<FFmpegDemuxer>(filepath_.c_str());

    if (verbose_.get()) {
      HOLOSCAN_LOG_INFO("---- Video File Info ----");
      HOLOSCAN_LOG_INFO("File: {}", filepath_);
      HOLOSCAN_LOG_INFO("Codec: {}", static_cast<int>(demuxer_->GetVideoCodec()));
      HOLOSCAN_LOG_INFO("Resolution: {}x{}", demuxer_->GetWidth(), demuxer_->GetHeight());
      HOLOSCAN_LOG_INFO("Bit Depth: {}", demuxer_->GetBitDepth());
      HOLOSCAN_LOG_INFO("Chroma Format: {}", static_cast<int>(demuxer_->GetChromaFormat()));
      HOLOSCAN_LOG_INFO("-------------------------");
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to initialize video demuxer: " + std::string(e.what()));
  }

  frame_count_ = 0;
  end_of_file_ = false;
}

void NvVideoReaderOp::compute(InputContext& op_input, OutputContext& op_output,
                              ExecutionContext& context) {
  auto enter_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                             std::chrono::steady_clock::now().time_since_epoch())
                             .count();

  // Check if we've reached end of file
  if (end_of_file_) {
    if (loop_.get()) {
      // Restart the demuxer
      try {
        demuxer_ = std::make_unique<FFmpegDemuxer>(filepath_.c_str());
        end_of_file_ = false;
        frame_count_ = 0;
        HOLOSCAN_LOG_INFO("Looping video file: {}", filepath_);
      } catch (const std::exception& e) {
        HOLOSCAN_LOG_ERROR("Failed to restart video demuxer: {}", e.what());
        return;
      }
    } else {
      // End of file reached and not looping
      return;
    }
  }

  // Get allocator for output tensor
  auto allocator =
      nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(), allocator_->gxf_cid());
  if (!allocator) {
    throw std::runtime_error("Failed to get allocator");
  }

  // Read video frames, skipping non-video packets
  uint8_t* frame_data = nullptr;
  int frame_size = 0;
  int64_t pts = 0;
  int64_t dts = 0;
  int is_video_packet = 0;
  bool success = false;

  // Keep reading until we get a video packet or reach end of file
  do {
    success = demuxer_->Demux(&frame_data, &frame_size, &pts, &dts, &is_video_packet);

    if (!success || frame_size == 0) {
      // End of file reached
      end_of_file_ = true;
      if (verbose_.get()) {
        HOLOSCAN_LOG_INFO("End of video file reached. Total frames read: {}", frame_count_);
      }
      return;
    }

    // Skip non-video packets and continue reading
    if (!is_video_packet) {
      if (verbose_.get()) {
        HOLOSCAN_LOG_DEBUG("Skipping non-video packet");
      }
      continue;
    }

    // Found a video packet, break out of loop
    break;
  } while (true);

  // Create output entity and tensor
  auto output_entity = nvidia::gxf::Entity::New(context.context());
  if (!output_entity) {
    throw std::runtime_error("Failed to create output entity");
  }

  auto tensor = output_entity.value().add<nvidia::gxf::Tensor>("");
  if (!tensor) {
    throw std::runtime_error("Failed to add tensor to output entity");
  }

  // Allocate tensor for the encoded frame data
  auto tensor_shape = nvidia::gxf::Shape{static_cast<int32_t>(frame_size)};
  auto result = tensor.value()->reshape<uint8_t>(
      tensor_shape, nvidia::gxf::MemoryStorageType::kHost, allocator.value());

  if (!result) {
    throw std::runtime_error("Failed to allocate tensor for encoded frame");
  }

  // Copy the encoded frame data
  std::memcpy(tensor.value()->pointer(), frame_data, frame_size);

  // Add timestamp information
  auto timestamp = output_entity.value().add<nvidia::gxf::Timestamp>("timestamp");
  if (timestamp) {
    timestamp.value()->acqtime = enter_timestamp;
    timestamp.value()->pubtime = enter_timestamp;
  }

  frame_count_++;

  auto emit_timestamp = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();

  // Add metadata
  auto meta = metadata();
  auto read_latency_ms = (emit_timestamp - enter_timestamp) / 1000000.0;
  meta->set("video_reader_read_latency_ms", read_latency_ms);
  meta->set("frame_number", frame_count_);
  meta->set("frame_size_bytes", frame_size);
  meta->set("source", std::string("nv_video_reader"));
  meta->set("codec", demuxer_->GetVideoCodec());
  meta->set("pts", pts);
  meta->set("dts", dts);
  meta->set("is_looping", loop_.get());

  // Signal stream reset to decoder when looping
  if (frame_count_ == 1) {
    meta->set("stream_reset", true);
    if (verbose_.get()) {
      HOLOSCAN_LOG_INFO("Signaling stream reset to decoder (first frame after loop or start)");
    }
  } else {
    meta->set("stream_reset", false);
  }

  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("Frame {} - PTS: {}, DTS: {}, Size: {} bytes, Video packet: {}",
                      frame_count_,
                      pts,
                      dts,
                      frame_size,
                      is_video_packet);
  }

  // Emit the output
  auto result_entity = gxf::Entity(std::move(output_entity.value()));
  op_output.emit(result_entity, "output");
}

void NvVideoReaderOp::stop() {
  if (demuxer_) {
    demuxer_.reset();
  }

  if (verbose_.get()) {
    HOLOSCAN_LOG_INFO("Video reader stopped. Total frames read: {}", frame_count_);
  }
}

}  // namespace holoscan::ops
