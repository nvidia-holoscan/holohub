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

#include "publisher.h"

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>

#include "operators/ucxx_send_receive/ucxx_endpoint.hpp"
#include "operators/ucxx_send_receive/ucxx_sender_op.hpp"

namespace holoscan::apps {

void UcxxEndoscopyPublisherApp::compose() {
  using namespace holoscan;

  HOLOSCAN_LOG_INFO("Composing PUBLISHER with full processing pipeline");

  // Constants for video dimensions
  const uint32_t width = 854;
  const uint32_t height = 480;
  const uint64_t source_block_size = width * height * 3 * 4;
  const uint64_t source_num_blocks = 2;

  // Create allocators
  auto replayer_allocator = make_resource<RMMAllocator>(
      "video_replayer_allocator",
                                      Arg("device_memory_max_size") = std::string("256MB"),
                                      Arg("device_memory_initial_size") = std::string("256MB"));

  auto cuda_stream_pool = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

  // Video replayer source
  auto replayer = make_operator<ops::VideoStreamReplayerOp>(
      "replayer",
      Arg("allocator") = replayer_allocator,
      Arg("directory", datapath_),
      from_config("replayer"));

  // Format converter for inference input
  auto format_converter = make_operator<ops::FormatConverterOp>(
      "format_converter",
      from_config("format_converter"),
      Arg("pool") = make_resource<BlockMemoryPool>(
          "pool", 1, source_block_size, source_num_blocks),
      Arg("cuda_stream_pool") = cuda_stream_pool);

  // LSTM inference
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

  // Tool tracking postprocessor
  const uint64_t tool_tracking_postprocessor_block_size =
      std::max(107 * 60 * 7 * 4 * sizeof(float), 7 * 3 * sizeof(float));
  const uint64_t tool_tracking_postprocessor_num_blocks = 2 * 2;

  auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
      "tool_tracking_postprocessor",
      Arg("device_allocator") = make_resource<BlockMemoryPool>(
          "device_allocator", 1, tool_tracking_postprocessor_block_size,
          tool_tracking_postprocessor_num_blocks));

  // Holoviz for visualization with render buffer output enabled for UCXX transmission
  const uint64_t render_buffer_size = width * height * 4 * 4;  // RGBA, 4 bytes per channel
  auto visualizer_allocator = make_resource<BlockMemoryPool>(
      "allocator", 1, render_buffer_size, source_num_blocks);

  auto holoviz = make_operator<ops::HolovizOp>(
      "holoviz",
      from_config("holoviz"),
      Arg("width") = width,
      Arg("height") = height,
      Arg("enable_render_buffer_output") = true,
      Arg("allocator") = visualizer_allocator,
      Arg("cuda_stream_pool") = cuda_stream_pool);

  // Format converter to convert HolovizOp's VideoBuffer output to Tensor for UCXX
  auto render_buffer_converter = make_operator<ops::FormatConverterOp>(
      "render_buffer_converter",
      Arg("in_dtype") = std::string("rgba8888"),
      Arg("out_dtype") = std::string("rgba8888"),
      Arg("out_tensor_name") = std::string(""),
      Arg("pool") = make_resource<BlockMemoryPool>(
          "pool", 1, render_buffer_size, source_num_blocks),
      Arg("cuda_stream_pool") = cuda_stream_pool);

  // UCXX endpoint for broadcasting to subscribers
  auto ucxx_endpoint = make_resource<holoscan::ops::UcxxEndpoint>(
      "ucxx_endpoint",
      Arg("hostname", hostname_),
      Arg("port", port_),
      Arg("listen", true));

  // UCXX sender to broadcast rendered frames
  auto ucxx_sender = make_operator<holoscan::ops::UcxxSenderOp>(
      "ucxx_sender",
      Arg("tag", 1ul),
      Arg("endpoint") = ucxx_endpoint,
      Arg("allocator") = replayer_allocator);

  // Build the pipeline
  add_flow(replayer, format_converter, {{"output", "source_video"}});
  add_flow(format_converter, lstm_inferer);
  add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
  add_flow(tool_tracking_postprocessor, holoviz, {{"out", "receivers"}});
  add_flow(replayer, holoviz, {{"output", "receivers"}});

  // Convert HolovizOp's VideoBuffer output to Tensor for UCXX transmission
  add_flow(holoviz, render_buffer_converter, {{"render_buffer_output", "source_video"}});
  add_flow(render_buffer_converter, ucxx_sender, {{"", "in"}});

  HOLOSCAN_LOG_INFO("Publisher pipeline: Replayer → Format → LSTM → Postprocess → "
                    "Holoviz → Convert → Broadcast");
}

}  // namespace holoscan::apps
