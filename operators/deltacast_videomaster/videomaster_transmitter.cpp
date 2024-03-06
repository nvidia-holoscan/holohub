/*
 * SPDX-FileCopyrightText:  Copyright (c) 2022, DELTACAST.TV. All rights reserved.
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

#include "videomaster_transmitter.hpp"

#include "holoscan/core/gxf/entity.hpp"
#include "holoscan/core/operator_spec.hpp"

#include "holoscan/core/resources/gxf/allocator.hpp"
#include "holoscan/core/resources/gxf/cuda_stream_pool.hpp"

namespace holoscan::ops {

void VideoMasterTransmitterOp::setup(OperatorSpec& spec) {
  auto& source = spec.input<gxf::Entity>("source");

  spec.param(_use_rdma, "rdma", "Use RDMA", "Specifies whether RDMA should be used.", false);
  // DEV FLEX
  spec.param(_board_index, "board", "Board", "Index of the Deltacast.TV board to use.", 0u);
  spec.param(_channel_index, "output", "Output", "Index of the output channel to use.", 0u);
  spec.param(_width, "width", "Width", "Width of the video frames to send.", 1920u);
  spec.param(_height, "height", "Height", "Height of the video frames to send.", 1080u);
  spec.param(_progressive,
             "progressive",
             "Progressive",
             "Progressiveness of the video frames to send.",
             true);
  spec.param(_framerate, "framerate", "Framerate", "Framerate of the signal to generate.", 60u);
  spec.param(_source, "source", "Source", "Source data.", &source);
  spec.param(_pool, "pool", "Pool", "Pool to allocate the buffers.");
  spec.param(_overlay,
             "enable_overlay",
             "EnableOverlay",
             "Specifies whether the input buffers should be treated as overlay data.",
             false);
}

}  // namespace holoscan::ops
