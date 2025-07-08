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

#ifndef NV_VIDEO_READER_NV_VIDEO_READER_HPP
#define NV_VIDEO_READER_NV_VIDEO_READER_HPP

#include <filesystem>
#include <memory>
#include <string>

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

#include "FFmpegDemuxer.h"

namespace holoscan::ops {

/**
 * @brief Operator to read H.264/H.265 video files and emit raw encoded frames
 *
 * This operator reads video files using FFmpeg demuxer and emits one encoded frame
 * at a time. The frames are not decoded - they remain in their compressed format
 * for processing by downstream operators like nv_video_decoder.
 */
class NvVideoReaderOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NvVideoReaderOp)

  NvVideoReaderOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  Parameter<std::string> directory_;
  Parameter<std::string> filename_;
  Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
  Parameter<bool> loop_;
  Parameter<bool> verbose_;

  // FFmpeg demuxer for reading video files
  std::unique_ptr<FFmpegDemuxer> demuxer_;

  // Store the full filepath
  std::string filepath_;

  // Tracking variables
  uint32_t frame_count_ = 0;
  bool end_of_file_ = false;
};

}  // namespace holoscan::ops

#endif /* NV_VIDEO_READER_NV_VIDEO_READER_HPP */
