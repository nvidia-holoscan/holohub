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

#include "video_encoder_request.hpp"
#include "video_encoder_custom_params.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {

void VideoEncoderRequestOp::initialize() {
  register_converter<nvidia::gxf::EncoderInputFormat>();
  register_converter<nvidia::gxf::EncoderConfig>();

  holoscan::ops::GXFOperator::initialize();
}

void VideoEncoderRequestOp::setup(OperatorSpec& spec) {
  auto& input = spec.input<gxf::Entity>("input_frame");

  spec.param(input_frame_,
             "input_frame",
             "InputFrame",
             "Receiver to get the input frame",
             &input);
  spec.param(videoencoder_context_,
             "videoencoder_context",
             "VideoEncoderContext",
             "Encoder context Handle");
  spec.param(inbuf_storage_type_,
             "inbuf_storage_type",
             "Input Buffer Storage(memory) type",
             "Input Buffer storage type, 0:kHost, 1:kDevice", 1U);
  spec.param(codec_,
             "codec",
             "Video Codec to use",
             "Video codec,  0:H264, only H264 supported",
             0);
  spec.param(input_height_,
             "input_height",
             "Input frame height",
             "");
  spec.param(input_width_,
             "input_width",
             "Input image width",
             "");
  spec.param(input_format_,
             "input_format",
             "Input color format, nv12,nv24,yuv420planar",
             "Default: nv12",
             nvidia::gxf::EncoderInputFormat::kNV12);
  spec.param(profile_,
             "profile",
             "Encode profile",
             "0:Baseline Profile, 1: Main , 2: High",
             2);
  spec.param(hw_preset_type_,
             "hw_preset_type",
             "Encode hw preset type, select from 0 to 3",
             "hw preset",
             0);
  spec.param(level_,
             "level",
             "Video H264 level",
             "Maximum data rate and resolution, select from 0 to 14",
             14);
  spec.param(iframe_interval_,
             "iframe_interval",
             "I Frame Interval",
             "Interval between two I frames",
             30);
  spec.param(rate_control_mode_,
             "rate_control_mode",
             "Rate control mode, 0:CQP[RC off], 1:CBR, 2:VBR ",
             "Rate control mode",
             1);
  spec.param(qp_,
             "qp",
             "Encoder constant QP value",
             "cont qp",
             20U);
  spec.param(bitrate_,
             "bitrate",
             "Encoder bitrate",
             "Bitrate of the encoded stream, in bits per second",
             20000000);
  spec.param(framerate_,
             "framerate",
             "Frame Rate, FPS",
             "Frames per second",
             30);
  spec.param(config_,
             "config",
             "Preset of parameters, select from pframe_cqp, iframe_cqp, custom",
             "Preset of config",
             nvidia::gxf::EncoderConfig::kCustom);
}

}  // namespace holoscan::ops
