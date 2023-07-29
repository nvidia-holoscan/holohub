/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_REQUEST
#define HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_REQUEST

#include <memory>
#include <string>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "video_encoder_utils.hpp"
#include "../video_encoder_context/video_encoder_context.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to handle the input for encoding YUV frames to H264 bit stream.
 *
 * This wraps a GXF Codelet(`nvidia::gxf::VideoEncoderRequest`).
 */
class VideoEncoderRequestOp: public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(VideoEncoderRequestOp, holoscan::ops::GXFOperator)

  VideoEncoderRequestOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::gxf::VideoEncoderRequest";
  }

  void initialize() override;
  void setup(OperatorSpec& spec) override;

 private:
  Parameter<holoscan::IOSpec*> input_frame_;
  Parameter<uint32_t> inbuf_storage_type_;
  Parameter<std::shared_ptr<holoscan::AsynchronousCondition>> async_scheduling_conditon_;
  Parameter<std::shared_ptr<holoscan::ops::VideoEncoderContext>> videoencoder_context_;
  Parameter<int32_t> codec_;
  Parameter<uint32_t> input_height_;
  Parameter<uint32_t> input_width_;
  Parameter<nvidia::gxf::EncoderInputFormat> input_format_;
  Parameter<int32_t> profile_;
  Parameter<int32_t> bitrate_;
  Parameter<int32_t> framerate_;
  Parameter<uint32_t> qp_;
  Parameter<int32_t> hw_preset_type_;
  Parameter<int32_t> level_;
  Parameter<int32_t> iframe_interval_;
  Parameter<int32_t> rate_control_mode_;
  Parameter<nvidia::gxf::EncoderConfig> config_;
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_VIDEO_ENCODER_REQUEST_VIDEO_ENCODER_REQUEST
