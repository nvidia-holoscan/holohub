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

#ifndef HOLOSCAN_OPERATORS_VIDEO_ENCODER_VIDEO_ENCODER
#define HOLOSCAN_OPERATORS_VIDEO_ENCODER_VIDEO_ENCODER

#include <memory>
#include <string>

#include "holoscan/core/gxf/gxf_operator.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to encode YUV video frames to h264 bit stream.
 *
 * This wraps a GXF Codelet(`nvidia::gxf::VideoEncoder`).
 */
class VideoEncoderOp: public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(VideoEncoderOp, holoscan::ops::GXFOperator)

  VideoEncoderOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::gxf::VideoEncoder";
  }

  void setup(OperatorSpec& spec) override;

 private:
  // Encoder I/O related Parameters
  Parameter<holoscan::IOSpec*> input_frame_;
  Parameter<holoscan::IOSpec*> output_transmitter_;
  Parameter<std::shared_ptr<Allocator>> pool_;
  Parameter<uint32_t> outbuf_storage_type_;
  Parameter<uint32_t> inbuf_storage_type_;

  // Encoder Parameters
  Parameter<std::string> device_;
  Parameter<int32_t> codec_;
  Parameter<uint32_t> input_height_;
  Parameter<uint32_t> input_width_;
  Parameter<std::string> input_format_;
  Parameter<int32_t> profile_;
  Parameter<int32_t> bitrate_;
  Parameter<int32_t> framerate_;
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_VIDEO_ENCODER_VIDEO_ENCODER
