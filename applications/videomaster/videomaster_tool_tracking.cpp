/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <holoscan/std_ops.hpp>

#include "videomaster_source.hpp"

class App : public holoscan::Application {
 public:

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;

    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t source_block_size = 0;
    uint64_t source_num_blocks = 0;

    width = from_config("videomaster.width").as<uint32_t>();
    height = from_config("videomaster.height").as<uint32_t>();
    source = make_operator<ops::VideoMasterSourceOp>(
            "videomaster",
            from_config("videomaster"),
            Arg("pool") = make_resource<UnboundedAllocator>("pool"));
    source_block_size = width * height * 4 * 4;
    source_num_blocks = from_config("videomaster.rdma").as<bool>() ? 3 : 4;

    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              from_config("format_converter_videomaster"),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks));

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);


    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        from_config("tool_tracking_postprocessor"),
        Arg("device_allocator") = make_resource<UnboundedAllocator>("device_allocator"),
        Arg("host_allocator") = make_resource<UnboundedAllocator>("host_allocator"));


    auto output_format_converter =
      make_operator<ops::FormatConverterOp>("output_format_converter", from_config("output_format_converter"),
      Arg("pool") = make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));

    auto drop_alpha_channel_converter =
      make_operator<ops::FormatConverterOp>("drop_alpha_channel_converter", from_config("drop_alpha_channel_converter"),
      Arg("pool") = make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));


    std::shared_ptr<BlockMemoryPool> visualizer_allocator;
    std::shared_ptr<ops::HolovizOp> visualizer = make_operator<ops::HolovizOp>(
        "holoviz",
        Arg("holoviz"),
        Arg("width") = width,
        Arg("height") = height,
        Arg("enable_render_buffer_input") = false,
        Arg("enable_render_buffer_output") = false,
        Arg("allocator") = visualizer_allocator,
        Arg("cuda_stream_pool") = cuda_stream_pool);

    // Flow definition
     // Input video display
    add_flow(source, drop_alpha_channel_converter);
    add_flow(drop_alpha_channel_converter, output_format_converter);
    add_flow(output_format_converter, visualizer, {{"", "receivers"}});

    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
    add_flow(tool_tracking_postprocessor, visualizer, {{"out", "receivers"}});
    add_flow(source, format_converter);
    add_flow(format_converter, lstm_inferer);
  }
};

int main(int argc, char** argv) {
  holoscan::load_env_log_level();

  auto app = holoscan::make_application<App>();

  if (argc == 2) {
    app->config(argv[1]);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/videomaster_tool_tracking.yaml";
    app->config(config_path);
  }

  app->run();

  return 0;
}
