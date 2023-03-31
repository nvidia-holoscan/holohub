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

#include "video_read_bitstream.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"
#include "holoscan/core/resources/gxf/allocator.hpp"

namespace holoscan::ops {
void VideoReadBitstreamOp::setup(OperatorSpec& spec) {
  auto& output =
    spec.output<gxf::Entity>("output_transmitter")
    .condition(ConditionType::kDownstreamMessageAffordable);

  spec.param(output_transmitter_,
             "output_transmitter",
             "OutputTransmitter",
             "Camera image containing camera frame and metadata",
             &output);
  spec.param(input_file_path_,
             "input_file_path",
             "File Path",
             "Path to image file");
  spec.param(pool_,
             "pool",
             "Memory pool for allocating output data",
             "");
  spec.param(outbuf_storage_type_,
             "outbuf_storage_type",
             "Output Buffer Storage(memory) type",
             "The memory storage type used by this "
             "allocator. Can be kHost (0), kDevice (1)");
  spec.param(aud_nal_present_,
             "aud_nal_present",
             "Access Unit Delimiter NAL Flag", "");
}

}  // namespace holoscan::ops
