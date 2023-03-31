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
#include "video_write_bitstream.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

namespace holoscan::ops {
void VideoWriteBitstreamOp::setup(OperatorSpec& spec) {
  auto& data_receiver = spec.input<gxf::Entity>("data_receiver");

  spec.param(output_video_path_,
               "output_video_path",
               "Output File Path",
               "File path to output video",
               std::string(""));
  spec.param(frame_width_,
               "frame_width",
               "The width of the output video",
               "");
  spec.param(frame_height_,
               "frame_height",
               "The height of the output video",
               "");
  spec.param(inbuf_storage_type_,
               "inbuf_storage_type",
               "Input memory type, 0:host mem, 1:device mem",
               "");
  spec.param(data_receiver_,
               "data_receiver",
               "DataReceiver",
               "Receiver to get the data",
               &data_receiver);
  spec.param(input_crc_file_path_,
               "input_crc_file_path",
               "Output CRC File", "Path to CRC File",
               std::string(""));
}

}  // namespace holoscan::ops
