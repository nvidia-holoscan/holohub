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

#ifndef NV_VIDEO_DECODER_NV_VIDEO_DECODER_HPP
#define NV_VIDEO_DECODER_NV_VIDEO_DECODER_HPP

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include <cuda.h>
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

#include "FFmpegDemuxer.h"
#include "NvDecoder/NvDecoder.h"

namespace holoscan::ops {
/**
 * @brief Data provider for streaming input to FFmpegDemuxer.
 *
 * StreamDataProvider implements the FFmpegDemuxer::DataProvider interface to allow
 * feeding arbitrary chunks of encoded video data (e.g., from a network stream or
 * custom source) into the demuxer/decoder. It accumulates incoming data in an
 * internal buffer and provides it to the demuxer as requested.
 *
 * Usage:
 *   - Call SetData() to append new encoded data to the buffer.
 *   - The demuxer will call GetData() to retrieve data for decoding.
 *   - Optionally, call ClearBuffer() to reset the buffer.
 */
class StreamDataProvider : public FFmpegDemuxer::DataProvider {
 public:
  StreamDataProvider() {
    // Pre-allocate buffer for accumulated packets
    buffer_.reserve(16 * 1024 * 1024);  // 16MB initial capacity
  }
  ~StreamDataProvider() {}

  void SetData(uint8_t* data, int size) {
    if (data && size > 0) {
      // Accumulate data instead of replacing it
      size_t old_size = buffer_.size();
      buffer_.resize(old_size + size);
      std::memcpy(buffer_.data() + old_size, data, size);

      // Reset offset if this is the first data or if we were at EOF
      if (old_size == 0 || offset_ >= old_size) {
        offset_ = old_size;  // Continue from where we left off
      }
    }
  }

  // Fill in the buffer owned by the demuxer/decoder and advance the offset
  int GetData(uint8_t* pBuf, int nBuf) {
    if (buffer_.empty() || offset_ >= buffer_.size()) {
      return AVERROR_EOF;
    }

    // Calculate how much data we can copy
    int remaining = buffer_.size() - offset_;
    int copy_size = std::min(nBuf, remaining);

    if (copy_size <= 0) {
      return AVERROR_EOF;
    }

    // Copy the data
    std::memcpy(pBuf, buffer_.data() + offset_, copy_size);
    offset_ += copy_size;

    return copy_size;
  }

  // Clear the buffer when we want to start fresh (optional utility method)
  void ClearBuffer() {
    buffer_.clear();
    offset_ = 0;
  }

  // Get current buffer size for debugging
  size_t GetBufferSize() const { return buffer_.size(); }
  size_t GetOffset() const { return offset_; }

 private:
  std::vector<uint8_t> buffer_;  // Accumulated packet data
  size_t offset_ = 0;            // Current read position
};

/**
 * @brief Operator to decode video frames using NVIDIA Video Codec SDK
 *
 * This operator takes video frames as input and decodes them to H264 format.
 * The input and output data remain on the GPU for maximum performance.
 */
class NvVideoDecoderOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NvVideoDecoderOp)

  NvVideoDecoderOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  void init_decoder_for_streaming(void* data, size_t size);
  void init_decoder_for_file(std::shared_ptr<MetadataDictionary> meta);

  Parameter<int> cuda_device_ordinal_;
  Parameter<int> width_;
  Parameter<int> height_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<bool> verbose_;

  CudaStreamHandler cuda_stream_handler_;

  // CUDA
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_;

  std::unique_ptr<NvDecoder> decoder_;
  std::unique_ptr<FFmpegDemuxer> demuxer_;
  std::unique_ptr<StreamDataProvider> file_data_provider_;

  uint64_t last_emit_timestamp_ = 0;
};

}  // namespace holoscan::ops

#endif /* NV_VIDEO_DECODER_NV_VIDEO_DECODER_HPP */
