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

#ifndef HOLOSCAN_OPERATORS_NV_VIDEO_DECODER_HPP
#define HOLOSCAN_OPERATORS_NV_VIDEO_DECODER_HPP

#include <memory>
#include <string>

#include <cuda.h>
#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/utils/cuda_stream_handler.hpp"

#include "../api/FFmpegDemuxer.h"
#include "../api/NvDecoder/NvDecoder.h"

namespace holoscan::ops {
class StreamDataProvider : public FFmpegDemuxer::DataProvider {
 public:
  StreamDataProvider() {}
  ~StreamDataProvider() {}

  void SetData(uint8_t* data, int size) {
    offset_ = 0;
    data_ = data;
    size_ = size;
  }
  // Fill in the buffer owned by the demuxer/decoder
  int GetData(uint8_t* pBuf, int nBuf) {
    if (!data_ || size_ == 0) {
      return AVERROR_EOF;
    }

    if (offset_ >= size_) {
      return AVERROR_EOF;
    }

    // Calculate how much data we can copy
    int remaining = size_ - offset_;
    int copy_size = std::min(nBuf, remaining);

    if (copy_size <= 0) {
      return AVERROR_EOF;
    }

    // Copy the data
    memcpy(pBuf, data_ + offset_, copy_size);
    offset_ += copy_size;

    // If we've consumed all the data, reset for next time
    if (offset_ >= size_) {
      data_ = nullptr;
      size_ = 0;
      offset_ = 0;
    }

    return copy_size;
  }

 private:
  uint8_t* data_ = nullptr;
  int size_ = 0;
  int offset_ = 0;
};

/**
 * @brief Operator to encode video frames using NVIDIA Video Codec SDK
 *
 * This operator takes video frames as input and encodes them to H264 format.
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
  void emit_empty_frame(OutputContext& op_output, ExecutionContext& context);
  Parameter<int> cuda_device_ordinal_;
  Parameter<int> width_;
  Parameter<int> height_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<bool> verbose_;

  CudaStreamHandler cuda_stream_handler_;

  // CUDA
  CUcontext cu_context_ = nullptr;
  CUdevice cu_device_;

  // NVIDIA Video Decoder
  std::unique_ptr<NvDecoder> decoder_;
  std::unique_ptr<FFmpegDemuxer> demuxer_;
  std::unique_ptr<StreamDataProvider> file_data_provider_;

  uint64_t last_emit_timestamp_ = 0;
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_NV_VIDEO_ENCODER_HPP
