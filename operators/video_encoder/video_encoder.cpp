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

#include "video_encoder.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {

constexpr char kDefaultDevice[] = "/dev/video0";

void VideoEncoderOp::setup(OperatorSpec& spec) {
  auto& input = spec.input<gxf::Entity>("input_frame");
  auto& output = spec.output<gxf::Entity>("output_transmitter");
  // I/O related parameters
spec.param(input_frame_,
           "input_frame",
           "InputFrame",
           "Receiver to get the input frame",
           &input);
spec.param(output_transmitter_,
           "output_transmitter",
           "OutputTransmitter",
           "Transmitter to send the compressed data",
           &output);
spec.param(pool_,
           "pool",
           "Memory pool for allocating output data",
           "");
spec.param(inbuf_storage_type_,
           "inbuf_storage_type",
           "Input Buffer storage(memory) type",
           "Input Buffer storage type, 0:host mem, 1:device mem");
spec.param(outbuf_storage_type_,
           "outbuf_storage_type",
           "Output Buffer storage(memory) type",
           "Output Buffer storage type, 0:host mem, 1:device mem");

// Encoder related parameters
spec.param(device_,
           "device", "VideoDevice",
           "Path to the V4L2 device",
           std::string(kDefaultDevice));
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
           "Input frame color format, nv12 PL is supported",
           "nv12pl");
spec.param(profile_,
           "profile", "Encode profile",
           "0:Baseline Profile, 1: Main , 2: High");
spec.param(bitrate_,
           "bitrate", "Encoder bitrate",
           "Bitrate of the encoded stream, in bits per second",
           20000000);
spec.param(framerate_,
           "framerate", "Frame Rate, FPS",
           "Frames per second",
           30);
}

}  // namespace holoscan::ops
