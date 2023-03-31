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

#include "video_decoder.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {

constexpr char kDefaultDevice[] = "/dev/nvidia0";

void VideoDecoderOp::setup(OperatorSpec& spec) {
  auto& input =
    spec.input<gxf::Entity>("image_receiver").condition(ConditionType::kMessageAvailable);
  auto& output = spec.output<gxf::Entity>("output_transmitter");

  // I/O related parameters
  spec.param(image_receiver_,
             "image_receiver",
             "ImageReceiver",
             "Receiver to get the input image",
             &input);
  spec.param(output_transmitter_,
             "output_transmitter",
             "OutputTransmitter",
             "Transmitter to send the compressed data",
             &output);
  spec.param(pool_,
             "pool",
             "Memory pool for allocating output data");
  spec.param(inbuf_storage_type_,
             "inbuf_storage_type",
             "Input Buffer Storage(memory) type",
             "Input Buffer storage type, 0:kHost, 1:kDevice");
  spec.param(outbuf_storage_type_,
             "outbuf_storage_type",
             "Output Buffer Storage(memory) type",
             "The memory storage type used by this "
             "allocator. Can be kHost (0), kDevice (1)");

  // Decoder related parameters
  spec.param(device_,
             "device", "VideoDevice",
             "Path to the V4L2 device",
             std::string(kDefaultDevice));
  spec.param(codec_,
             "codec",
             "Video Codec to use",
             "Video codec,  0:H264, only H264 supported",
             0);
}

}  // namespace holoscan::ops
