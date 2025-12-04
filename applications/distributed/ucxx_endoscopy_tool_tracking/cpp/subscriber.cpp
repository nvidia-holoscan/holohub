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

#include "subscriber.h"

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/video_stream_recorder/video_stream_recorder.hpp>

#include "operators/ucxx_send_receive/ucxx_endpoint.hpp"
#include "operators/ucxx_send_receive/ucxx_receiver_op.hpp"

namespace holoscan::apps {

void UcxxEndoscopySubscriberApp::compose() {
  using namespace holoscan;

  const uint32_t width = 854;
  const uint32_t height = 480;
  const uint64_t source_block_size = width * height * 4 * 4;
  const uint64_t source_num_blocks = 2;

  HOLOSCAN_LOG_INFO("Composing SUBSCRIBER - receiving and displaying rendered frames");

  auto allocator = make_resource<RMMAllocator>("video_replayer_allocator",
                                      Arg("device_memory_max_size") = std::string("256MB"),
                                      Arg("device_memory_initial_size") = std::string("256MB"));

  // UCXX endpoint for receiving from publisher
  auto ucxx_endpoint = make_resource<holoscan::ops::UcxxEndpoint>(
      "ucxx_endpoint",
      Arg("hostname", hostname_),
      Arg("port", port_),
      Arg("listen", false));

  // UCXX receiver to get rendered frames from publisher as Tensor
  // Buffer size for RGBA frame: width * height * 4 channels + metadata overhead
  const int buffer_size = (4 << 10) + width * height * 4;
  auto ucxx_receiver = make_operator<holoscan::ops::UcxxReceiverOp>(
      "ucxx_receiver",
      Arg("tag", 1ul),
      Arg("buffer_size", buffer_size),
      Arg("endpoint") = ucxx_endpoint,
      Arg("allocator") = allocator);

  // Subscriber-side visualization - simple image display
  auto holoviz = make_operator<ops::HolovizOp>(
      "holoviz",
      from_config("holoviz_subscriber"),
      Arg("width") = width,
      Arg("height") = height,
      Arg("allocator") = allocator);

  // Optional recorder for testing/validation (enabled based on record_type parameter)
  std::shared_ptr<Operator> recorder;
  std::shared_ptr<Operator> recorder_format_converter;
  bool enable_recording = false;

  // Check record_type to determine if recording should be enabled
  auto record_type = from_config("record_type").as<std::string>();
  if (record_type == "subscriber") {
    enable_recording = true;
    HOLOSCAN_LOG_INFO("Subscriber recording enabled (record_type=subscriber)");

    recorder_format_converter = make_operator<ops::FormatConverterOp>(
        "recorder_format_converter",
        from_config("recorder_format_converter"),
        Arg("pool") = make_resource<BlockMemoryPool>(
            "pool", 1, source_block_size, source_num_blocks));

    recorder = make_operator<ops::VideoStreamRecorderOp>(
        "recorder_subscriber",
        from_config("recorder_subscriber"));
  } else {
    HOLOSCAN_LOG_INFO("Subscriber recording disabled (record_type={})", record_type);
  }

  // Display received rendered frames (Tensor output)
  add_flow(ucxx_receiver, holoviz, {{"out", "receivers"}});

  // Optional recording of received frames for testing
  if (enable_recording) {
    add_flow(ucxx_receiver, recorder_format_converter, {{"out", "source_video"}});
    add_flow(recorder_format_converter, recorder);
  }

  HOLOSCAN_LOG_INFO("Subscriber pipeline: Receive → Display{}",
                   enable_recording ? " + Record" : "");
}

}  // namespace holoscan::apps

