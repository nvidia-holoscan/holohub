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

#include <getopt.h>

#include "holoscan/holoscan.hpp"
#include <holoscan/operators/aja_source/aja_source.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/video_stream_recorder/video_stream_recorder.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) {
    if (source == "aja") { is_aja_source_ = true; }
  }

  enum class Record { NONE, INPUT, VISUALIZER };

  void set_record(const std::string& record) {
    if (record == "input") {
      record_type_ = Record::INPUT;
    } else if (record == "visualizer") {
      record_type_ = Record::VISUALIZER;
    }
  }

  void set_datapath(const std::string& path) {
     datapath = path;
  }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;
    std::shared_ptr<Operator> recorder;
    std::shared_ptr<Operator> recorder_format_converter;

    const bool is_rdma = from_config("aja.rdma").as<bool>();
    const bool is_aja_overlay_enabled =
        is_aja_source_ && from_config("aja.enable_overlay").as<bool>();
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t source_block_size = 0;
    uint64_t source_num_blocks = 0;

    if (is_aja_source_) {
      width = from_config("aja.width").as<uint32_t>();
      height = from_config("aja.height").as<uint32_t>();
      source = make_operator<ops::AJASourceOp>("aja", from_config("aja"));
      source_block_size = width * height * 4 * 4;
      source_num_blocks = is_rdma ? 3 : 4;
    } else {
      width = 854;
      height = 480;
      source = make_operator<ops::VideoStreamReplayerOp>(
          "replayer", from_config("replayer"), Arg("directory", datapath));
      source_block_size = width * height * 3 * 4;
      source_num_blocks = 2;
    }

    if (record_type_ != Record::NONE) {
      if (((record_type_ == Record::INPUT) && is_aja_source_) ||
          (record_type_ == Record::VISUALIZER)) {
        recorder_format_converter = make_operator<ops::FormatConverterOp>(
            "recorder_format_converter",
            from_config("recorder_format_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
      }
      recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));
    }

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter",
        from_config(is_aja_source_ ? "format_converter_aja" : "format_converter_replayer"),
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
    if ((record_type_ == Record::VISUALIZER) && !is_aja_source_) {
      visualizer_allocator =
          make_resource<BlockMemoryPool>("allocator", 1, source_block_size, source_num_blocks);
    }
    std::shared_ptr<ops::HolovizOp> visualizer = make_operator<ops::HolovizOp>(
        "holoviz",
        from_config(is_aja_overlay_enabled ? "holoviz_overlay" : "holoviz"),
        Arg("width") = width,
        Arg("height") = height,
        Arg("enable_render_buffer_input") = is_aja_overlay_enabled,
        Arg("enable_render_buffer_output") =
            is_aja_overlay_enabled || (record_type_ == Record::VISUALIZER),
        Arg("allocator") = visualizer_allocator,
        Arg("cuda_stream_pool") = cuda_stream_pool);

    // Flow definition
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
    add_flow(tool_tracking_postprocessor, visualizer, {{"out", "receivers"}});

    add_flow(source,
             format_converter,
             {{is_aja_source_ ? "video_buffer_output" : "output", "source_video"}});
    add_flow(format_converter, lstm_inferer);

    if (is_aja_overlay_enabled) {
      // Overlay buffer flow between AJA source and visualizer
      add_flow(source, visualizer, {{"overlay_buffer_output", "render_buffer_input"}});
      add_flow(visualizer, source, {{"render_buffer_output", "overlay_buffer_input"}});
    } else {
      add_flow(
          source, visualizer, {{is_aja_source_ ? "video_buffer_output" : "output", "receivers"}});
    }

    if (record_type_ == Record::INPUT) {
      if (is_aja_source_) {
        add_flow(source, recorder_format_converter, {{"video_buffer_output", "source_video"}});
        add_flow(recorder_format_converter, recorder);
      } else {
        add_flow(source, recorder);
      }
    } else if (record_type_ == Record::VISUALIZER) {
      add_flow(visualizer, recorder_format_converter, {{"render_buffer_output", "source_video"}});
      add_flow(recorder_format_converter, recorder);
    }
  }

 private:
  bool is_aja_source_ = false;
  Record record_type_ = Record::NONE;
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

/** Main function */
int main(int argc, char** argv) {
  holoscan::load_env_log_level();

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
    config_path += "/endoscopy_tool_tracking.yaml";
    app->config(config_path);
  }
  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);
  auto record_type = app->from_config("record_type").as<std::string>();
  app->set_record(record_type);

  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
