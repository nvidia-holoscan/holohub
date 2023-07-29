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

#include "video_decoder_request.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

namespace holoscan::ops {

constexpr char kDefaultOutputFormat[] = "nv12pl";

void VideoDecoderRequestOp::setup(OperatorSpec& spec) {
  auto& input = spec.input<gxf::Entity>("input_frame");

  // I/O related parameters
  spec.param(input_frame_,
             "input_frame",
             "InputFrame",
             "Receiver to get the input image",
             &input);
  spec.param(inbuf_storage_type_,
             "inbuf_storage_type",
             "Input Buffer Storage(memory) type",
             "Input Buffer storage type, 0:kHost, 1:kDevice");
  spec.param(async_scheduling_term_,
             "async_scheduling_term",
             "Asynchronous Scheduling Condition",
             "Asynchronous Scheduling Condition");
  spec.param(videodecoder_context_,
             "videodecoder_context",
             "VideoDecoderContext",
             "Decoder context Handle");
  spec.param(codec_,
             "codec",
             "Video Codec to use",
             "Video codec,  0:H264, only H264 supported",
             0U);
  spec.param(disableDPB_,
             "disableDPB",
             "Enable low latency decode",
             "Works only for IPPP case",
             0U);
  spec.param(output_format_,
             "output_format",
             "Output frame video format",
             "nv12pl and yuv420planar are supported",
             std::string(kDefaultOutputFormat));
}

}  // namespace holoscan::ops
