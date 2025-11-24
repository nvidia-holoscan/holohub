/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "server.h"

#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>

#include "operators/ucxx_send_receive/ucxx_endpoint.h"
#include "operators/ucxx_send_receive/ucxx_sender_op.h"

namespace holoscan::apps {

void UcxxEndoscopyServerApp::compose() {
  using namespace holoscan;

  HOLOSCAN_LOG_INFO("Composing SIMPLIFIED SERVER - replaying and broadcasting raw frames");

  // Video replayer source
  auto replayer = make_operator<ops::VideoStreamReplayerOp>(
      "replayer",
      from_config("replayer"),
      Arg("directory", datapath_));

  // UCXX endpoint for broadcasting to clients
  auto ucxx_endpoint = make_resource<isaac::os::UcxxEndpoint>(
      "ucxx_endpoint",
      Arg("hostname", hostname_),
      Arg("port", port_),
      Arg("listen", true));

  // UCXX sender to broadcast raw frames as Tensor
  auto ucxx_sender = make_operator<isaac::os::ops::UcxxSenderOp>(
      "ucxx_sender",
      Arg("tag", 1ul),
      Arg("endpoint", ucxx_endpoint));

  // Simple pipeline: Replayer → UCXX Sender
  add_flow(replayer, ucxx_sender, {{"output", "in"}});

  HOLOSCAN_LOG_INFO("Simplified server pipeline: Replayer → Broadcast");
}

}  // namespace holoscan::apps
