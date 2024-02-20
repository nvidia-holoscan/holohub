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

#ifndef HOLOSCAN_OPERATORS_VIDEO_DECODER_RESPONSE_VIDEO_DECODER_RESPONSE
#define HOLOSCAN_OPERATORS_VIDEO_DECODER_RESPONSE_VIDEO_DECODER_RESPONSE

#include <memory>
#include <string>

#include "holoscan/core/gxf/gxf_operator.hpp"
#include "../video_decoder_context/video_decoder_context.hpp"

namespace holoscan::ops {

/**
 * @brief Operator class to handle the output of the decoded H264 bit stream..
 *
 * This wraps a GXF Codelet(`nvidia::gxf::VideoDecoderResponse`).
 */
class VideoDecoderResponseOp: public holoscan::ops::GXFOperator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS_SUPER(VideoDecoderResponseOp,
      holoscan::ops::GXFOperator)

  VideoDecoderResponseOp() = default;

  const char* gxf_typename() const override {
    return "nvidia::gxf::VideoDecoderResponse";
  }

  void setup(OperatorSpec& spec) override;

 private:
  Parameter<holoscan::IOSpec*> output_transmitter_;
  Parameter<std::shared_ptr<Allocator>> pool_;
  Parameter<uint32_t> outbuf_storage_type_;
  Parameter<std::shared_ptr<holoscan::ops::VideoDecoderContext>>
      videodecoder_context_;
};

}  // namespace holoscan::ops

#endif  // HOLOSCAN_OPERATORS_VIDEO_DECODER_RESPONSE_VIDEO_DECODER_RESPONSE
