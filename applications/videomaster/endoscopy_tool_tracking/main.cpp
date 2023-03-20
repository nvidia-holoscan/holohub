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

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    const bool is_overlay_enabled = from_config("videomaster.overlay").as<bool>();

    uint32_t width = from_config("videomaster.width").as<uint32_t>();
    uint32_t height = from_config("videomaster.height").as<uint32_t>();
    std::shared_ptr<Operator> source = make_operator<ops::VideoMasterSourceOp>(
        "videomaster",
        from_config("videomaster"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));
    uint64_t source_block_size = width * height * 4 * 4;
    uint64_t source_num_blocks = from_config("videomaster.use_rdma").as<bool>() ? 3 : 4;

    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              from_config("format_converter"),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks));

    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));

    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        from_config("tool_tracking_postprocessor"),
        Arg("device_allocator") = make_resource<UnboundedAllocator>("device_allocator"),
        Arg("host_allocator") = make_resource<UnboundedAllocator>("host_allocator"));

    std::shared_ptr<BlockMemoryPool> visualizer_allocator;
    std::shared_ptr<ops::HolovizOp> visualizer = make_operator<ops::HolovizOp>(
        "holoviz",
        from_config(is_overlay_enabled ? "holoviz_overlay" : "holoviz"),
        Arg("width") = width,
        Arg("height") = height,
        Arg("enable_render_buffer_output") = is_overlay_enabled,
        Arg("allocator") = visualizer_allocator);

    // Flow definition
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
    add_flow(tool_tracking_postprocessor, visualizer, {{"out", "receivers"}});
    add_flow(source, format_converter, {{"signal", "source_video"}});
    add_flow(format_converter, lstm_inferer);

    if (is_overlay_enabled) {
      // Overlay buffer flow between source and visualizer
      auto overlayer = make_operator<ops::VideoMasterTransmitterOp>(
          "videomaster_overlayer",
          from_config("videomaster"),
          Arg("pool") = make_resource<UnboundedAllocator>("pool"));
      auto overlay_format_converter_videomaster = make_operator<ops::FormatConverterOp>(
          "overlay_format_converter",
          from_config("overlay_format_converter"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
      add_flow(visualizer, overlay_format_converter_videomaster, {{"render_buffer_output", ""}});
      add_flow(overlay_format_converter_videomaster, overlayer);
    } else {
      auto visualizer_format_converter_videomaster = make_operator<ops::FormatConverterOp>(
          "visualizer_format_converter",
          from_config("visualizer_format_converter"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
      auto drop_alpha_channel_converter = make_operator<ops::FormatConverterOp>(
          "drop_alpha_channel_converter",
          from_config("drop_alpha_channel_converter"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
      add_flow(source, drop_alpha_channel_converter);
      add_flow(drop_alpha_channel_converter, visualizer_format_converter_videomaster);
      add_flow(visualizer_format_converter_videomaster, visualizer, {{"", "receivers"}});
    }
  }
};

int main(int argc, char** argv) {
  holoscan::load_env_log_level();

  auto app = holoscan::make_application<App>();

  if (argc == 2) {
    app->config(argv[1]);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/app_config.yaml";
    app->config(config_path);
  }

  app->run();

  return 0;
}
