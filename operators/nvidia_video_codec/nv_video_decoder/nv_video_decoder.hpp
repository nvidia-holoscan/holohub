/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
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
 * @brief Operator to decode compressed video using the NVIDIA Video Codec SDK.
 *
 * By default the operator uses the existing demuxed/file streaming path. Setting
 * `codec` to `"H264"` or `"HEVC"` enables packetized input and feeds each input
 * tensor directly to the CUVID parser, bypassing the FFmpeg demuxer.
 * Packetized input tensors must use host-accessible `kHost` or `kSystem` storage;
 * device-backed encoded payloads are not accepted by the CUVID parser path.
 * Packetized bitstreams must decode to 8-bit 4:2:0 NV12; other decoded surface
 * formats are rejected instead of being copied into an incompatible NV12 buffer.
 *
 * `packetized_input_mode` describes the framing of that packetized input:
 *
 * - `"stream"` (default): input tensors are arbitrary byte-stream chunks. A picture
 *   may span multiple input tensors and CUVID determines picture boundaries.
 * - `"access_unit"`: every input tensor contains exactly one complete encoded access
 *   unit. The operator marks each submitted packet with `CUVID_PKT_ENDOFPICTURE`,
 *   allowing CUVID to complete the current picture without waiting for data from the
 *   next input tensor.
 *
 * `"access_unit"` must only be selected when the input contract guarantees one
 * complete access unit per tensor. Using it with fragmented input can cause incorrect
 * parser boundaries or decode failures.
 * With normal display latency, every access-unit tensor must also provide the picture's
 * presentation timestamp in the `presentation_timestamp_ns` metadata field. The value is
 * expressed in nanoseconds and is converted internally to the decoder timebase so metadata
 * remains associated with the correct picture when B-frames are reordered. Low-latency mode
 * can use a synthetic timestamp when this metadata field is absent.
 *
 * For a finite packetized stream, set the `end_of_stream` metadata field to `true`
 * on the final input tensor. The operator decodes that tensor, submits a distinct
 * end-of-stream packet to CUVID, and queues frames delayed by parser lookahead or
 * display reordering. Decoded frames are emitted one per operator execution so the
 * default output connector capacity is respected.
 *
 * `packetized_low_latency` independently controls the decoder display policy. Its
 * default value, `false`, preserves normal CUVID display reordering and supports
 * streams containing B-frames. Setting it to `true` reduces display delay and is
 * intended only for low-latency bitstreams without B-frames, such as All-Intra or
 * IPPP streams. It does not change input framing or end-of-picture signaling.
 *
 * Decoded NV12 frames are emitted in device memory.
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
  struct PendingFrame {
    uint8_t* data = nullptr;
    MetadataDictionary metadata;
    int64_t decode_start_timestamp = 0;
  };

  struct PendingAccessUnitMetadata {
    MetadataDictionary metadata;
    int64_t decode_start_timestamp = 0;
  };

  void emit_pending_frame(OutputContext& op_output, ExecutionContext& context);
  void release_pending_frames();
  void init_decoder_for_streaming(void* data, size_t size);
  void init_decoder_for_file(std::shared_ptr<MetadataDictionary> meta);
  void init_decoder_for_packetized_stream();

  Parameter<int> cuda_device_ordinal_;
  Parameter<int> width_;
  Parameter<int> height_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<bool> verbose_;
  Parameter<std::string> codec_;
  Parameter<std::string> packetized_input_mode_;
  Parameter<bool> packetized_low_latency_;

  CudaStreamHandler cuda_stream_handler_;

  // CUDA
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_;

  std::unique_ptr<NvDecoder> decoder_;
  std::unique_ptr<FFmpegDemuxer> demuxer_;
  std::unique_ptr<StreamDataProvider> file_data_provider_;
  std::deque<PendingFrame> pending_frames_;
  std::unordered_map<int64_t, PendingAccessUnitMetadata> pending_access_unit_metadata_;
  std::atomic<std::size_t> pending_frame_count_{0};
  std::shared_ptr<Condition> input_or_pending_condition_;

  int64_t next_access_unit_timestamp_ = 1;
  uint64_t last_emit_timestamp_ = 0;
};

}  // namespace holoscan::ops

#endif /* NV_VIDEO_DECODER_NV_VIDEO_DECODER_HPP */
