/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_VIDEO_WRITE_BITSTREAM_VIDEO_WRITE_BITSTREAM
#define HOLOSCAN_OPERATORS_VIDEO_WRITE_BITSTREAM_VIDEO_WRITE_BITSTREAM

#include <string>

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to write video bit stream.
 *
 * This wraps a GXF Codelet(`nvidia::gxf::VideoWriteBitstream`).
 */
class VideoWriteBitstreamOp: public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(VideoWriteBitstreamOp, holoscan::ops::GXFOperator)

  VideoWriteBitstreamOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::gxf::VideoWriteBitstream";
  }

  void setup(OperatorSpec& spec) override;

 private:
  // The path to save the video
  Parameter<std::string> output_video_path_;
  /// The width of the output video
  Parameter<int> frame_width_;
  // The height of the output video
  Parameter<int> frame_height_;
  // Input buffer storage type
  Parameter<int> inbuf_storage_type_;
  // Data receiver to get data
  Parameter<holoscan::IOSpec*> data_receiver_;
  // File for CRC verification
  Parameter<std::string> input_crc_file_path_;
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_VIDEO_WRITE_BITSTREAM_VIDEO_WRITE_BITSTREAM
