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

  const uint32_t width = 854;
  const uint32_t height = 480;
  const uint64_t source_block_size = width * height * 3 * 4;
  const uint64_t source_num_blocks = 2;

  HOLOSCAN_LOG_INFO("Composing SERVER application - processing and broadcasting frames");

  // Video replayer source
  auto replayer = make_operator<ops::VideoStreamReplayerOp>(
      "replayer",
      from_config("replayer"),
      Arg("directory", datapath_));

  // Create CUDA stream pool
  const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
      make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

  // Format converter
  auto format_converter = make_operator<ops::FormatConverterOp>(
      "format_converter",
      from_config("format_converter"),
      Arg("pool") = make_resource<BlockMemoryPool>(
          "pool", 1, source_block_size, source_num_blocks),
      Arg("cuda_stream_pool") = cuda_stream_pool);

  // LSTM TensorRT Inference
  const std::string model_file_path = datapath_ + "/tool_loc_convlstm.onnx";
  const std::string engine_cache_dir = datapath_ + "/engines";

  const uint64_t lstm_inferer_block_size = 107 * 60 * 7 * 4;
  const uint64_t lstm_inferer_num_blocks = 2 + 5 * 2;
  auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
      "lstm_inferer",
      from_config("lstm_inference"),
      Arg("model_file_path", model_file_path),
      Arg("engine_cache_dir", engine_cache_dir),
      Arg("pool") = make_resource<BlockMemoryPool>(
          "pool", 1, lstm_inferer_block_size, lstm_inferer_num_blocks),
      Arg("cuda_stream_pool") = cuda_stream_pool);

  // Tool tracking post-processor
  const uint64_t tool_tracking_postprocessor_block_size =
      std::max(107 * 60 * 7 * 4 * sizeof(float), 7 * 3 * sizeof(float));
  const uint64_t tool_tracking_postprocessor_num_blocks = 2 * 2;
  
  auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
      "tool_tracking_postprocessor",
      Arg("device_allocator") = make_resource<BlockMemoryPool>(
          "device_allocator",
          1,
          tool_tracking_postprocessor_block_size,
          tool_tracking_postprocessor_num_blocks));

  // Server-side visualization
  auto holoviz = make_operator<ops::HolovizOp>(
      "holoviz",
      from_config("holoviz"),
      Arg("width") = width,
      Arg("height") = height,
      Arg("cuda_stream_pool") = cuda_stream_pool);

  // UCXX endpoint for broadcasting to clients
  auto ucxx_endpoint = make_resource<isaac::os::UcxxEndpoint>(
      "ucxx_endpoint",
      Arg("hostname", hostname_),
      Arg("port", port_),
      Arg("listen", true));

  // UCXX sender to broadcast processed frames
  auto ucxx_sender = make_operator<isaac::os::ops::UcxxSenderOp>(
      "ucxx_sender",
      Arg("tag", 1ul),
      Arg("endpoint", ucxx_endpoint));

  // Build processing pipeline
  add_flow(replayer, format_converter, {{"output", "source_video"}});
  add_flow(format_converter, lstm_inferer);
  add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});

  // Display processed results locally
  add_flow(replayer, holoviz, {{"output", "receivers"}});
  add_flow(tool_tracking_postprocessor, holoviz, {{"out", "receivers"}});

  // Broadcast processed results to clients
  add_flow(tool_tracking_postprocessor, ucxx_sender);

  HOLOSCAN_LOG_INFO("Server pipeline: Replayer → Processing → Local Display + Broadcast");
}

}  // namespace holoscan::apps
