/*
 * SPDX-FileCopyrightText: Copyright (c) DELTACAST.TV. All rights reserved.
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

#include "holoscan/holoscan.hpp"
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/video_stream_recorder/video_stream_recorder.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <videomaster_source.hpp>
#include <videomaster_transmitter.hpp>

#include <getopt.h>

class App : public holoscan::Application {
 public:
  /** Sets the path to the data directory */
  void set_datapath(const std::string& path) {
    datapath = path;
  }

  /** Compose function */
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

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter",
        from_config("format_converter"),
        Arg("pool") =
            make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    const std::string model_file_path = datapath+"/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = datapath+"/engines";

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

    const uint64_t tool_tracking_postprocessor_block_size = 107 * 60 * 7 * 4;
    const uint64_t tool_tracking_postprocessor_num_blocks = 2;
    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
         Arg("device_allocator") =
            make_resource<BlockMemoryPool>("device_allocator",
                                           1,
                                           tool_tracking_postprocessor_block_size,
                                           tool_tracking_postprocessor_num_blocks),
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

 private:
  std::string datapath = "data/endoscopy";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {
      {"data",    required_argument, 0,  'd' },
      {0,         0,                 0,  0 }
  };

  while (int c = getopt_long(argc, argv, "d",
                   long_options, NULL))  {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) {
    config_name = argv[optind++];
  }
  return true;
}

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) {
    return 1;
  }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/deltacast_endoscopy_tool_tracking.yaml";
    app->config(config_path);
  }

  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
