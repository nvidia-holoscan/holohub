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

#include "client.h"

#include <holoscan/operators/holoviz/holoviz.hpp>

#include "operators/ucxx_send_receive/ucxx_endpoint.h"
#include "operators/ucxx_send_receive/ucxx_receiver_op.h"

namespace holoscan::apps {

void UcxxEndoscopyClientApp::compose() {
  using namespace holoscan;

  const uint32_t width = 854;
  const uint32_t height = 480;

  HOLOSCAN_LOG_INFO("Composing CLIENT application - receiving and displaying processed frames");

  // UCXX endpoint for receiving from server
  auto ucxx_endpoint = make_resource<isaac::os::UcxxEndpoint>(
      "ucxx_endpoint",
      Arg("hostname", hostname_),
      Arg("port", port_),
      Arg("listen", false));

  // UCXX receiver to get processed frames from server
  // Buffer size needs to accommodate the postprocessor output
  const uint64_t buffer_size = 107 * 60 * 7 * 4 * sizeof(float);
  auto ucxx_receiver = make_operator<isaac::os::ops::UcxxReceiverOp>(
      "ucxx_receiver",
      Arg("tag", 1ul),
      Arg("schema_name", "isaac.ToolTrackingPostprocessorOutput"),
      Arg("buffer_size", buffer_size),
      Arg("endpoint", ucxx_endpoint));

  // Client-side visualization
  auto holoviz = make_operator<ops::HolovizOp>(
      "holoviz",
      from_config("holoviz_client"),
      Arg("width") = width,
      Arg("height") = height);

  // Display received processed frames
  add_flow(ucxx_receiver, holoviz, {{"out", "receivers"}});

  HOLOSCAN_LOG_INFO("Client pipeline: Receive → Display");
}

}  // namespace holoscan::apps

