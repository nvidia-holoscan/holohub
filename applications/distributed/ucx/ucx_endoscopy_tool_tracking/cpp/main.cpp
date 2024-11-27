/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <iostream>
#include <lstm_tensor_rt_inference.hpp>
#include <string>
#include <tool_tracking_postprocessor.hpp>
#include "holoscan/holoscan.hpp"

using namespace holoscan;
class VideoInputFragment : public holoscan::Fragment {
 private:
  std::string input_dir_;

 public:
  explicit VideoInputFragment(const std::string& input_dir) : input_dir_(input_dir) {}

  void compose() override {
    ArgList args;
    args.add(Arg("directory", input_dir_));
    HOLOSCAN_LOG_INFO("Using video from {}", input_dir_);

    auto replayer = make_operator<ops::VideoStreamReplayerOp>(
        "replayer",
        Arg("allocator",
            make_resource<RMMAllocator>("video_replayer_allocator",
                                        Arg("device_memory_max_size") = std::string("256MB"),
                                        Arg("device_memory_initial_size") = std::string("256MB"))),
        from_config("replayer"),
        args);

    add_operator(replayer);
  }
};

class CloudInferenceFragment : public holoscan::Fragment {
 private:
  std::string model_dir_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint64_t source_block_size_ = 0;
  uint64_t source_num_blocks_ = 0;

 public:
  CloudInferenceFragment(const std::string& model_dir, const uint32_t width, const uint32_t height,
                         const uint64_t source_block_size, const uint64_t source_num_blocks)
      : model_dir_(model_dir),
        width_(width),
        height_(height),
        source_block_size_(source_block_size),
        source_num_blocks_(source_num_blocks) {}

  void compose() override {
    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter",
        from_config("format_converter"),
        Arg("pool") =
            make_resource<BlockMemoryPool>("pool", 1, source_block_size_, source_num_blocks_),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    const std::string model_file_path = model_dir_ + "/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = model_dir_ + "/engines";

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

    // Due to an underlying change in the GXF UCX extension in GXF 4.0 that results in a known issue
    // where we have to allocate more blocks than expected when using a BlockMemoryPool, we need to
    // use UnboundedAllocator for now.
    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        Arg("device_allocator") = make_resource<UnboundedAllocator>("device_allocator"));

    add_flow(format_converter, lstm_inferer);
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
  }
};

class VizFragment : public holoscan::Fragment {
 private:
  uint32_t width_ = 0;
  uint32_t height_ = 0;

 public:
  VizFragment(const uint32_t width, const uint32_t height) : width_(width), height_(height) {}

  void compose() override {
    std::shared_ptr<BlockMemoryPool> visualizer_allocator;

    auto visualizer_operator = make_operator<ops::HolovizOp>(
        "holoviz",
        from_config("holoviz"),
        Arg("width") = width_,
        Arg("height") = height_,
        Arg("allocator") = visualizer_allocator,
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));
    add_operator(visualizer_operator);
  }
};

class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath_ = path; }
  void compose() override {
    using namespace holoscan;

    auto width = 854;
    auto height = 480;
    auto source_block_size = width * height * 3 * 4;
    auto source_num_blocks = 2;

    auto video_in = make_fragment<VideoInputFragment>("video_in", datapath_);
    auto cloud_inference = make_fragment<CloudInferenceFragment>(
        "inference", datapath_, width, height, source_block_size, source_num_blocks);
    auto viz = make_fragment<VizFragment>("viz", width, height);

    // Flow definition
    add_flow(video_in, cloud_inference, {{"replayer", "format_converter"}});
    add_flow(cloud_inference, viz, {{"tool_tracking_postprocessor.out", "holoviz.receivers"}});

    add_flow(video_in, viz, {{"replayer.output", "holoviz.receivers"}});
  }

 private:
  std::string datapath_ = "data/endoscopy";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& data_path, std::string& config_path) {
  static struct option long_options[] = {
      {"data", required_argument, 0, 'd'}, {"config", required_argument, 0, 'c'}, {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d:c:", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'c':
        config_path = optarg;
        break;
      case 'd':
        data_path = optarg;
        break;
      default:
        holoscan::log_error("Unhandled option '{}'", static_cast<char>(c));
        return false;
    }
  }

  return true;
}

/** Main function */
int main(int argc, char** argv) {
  std::string config_path = "";
  std::string data_directory = "";
  if (!parse_arguments(argc, argv, data_directory, config_path)) { return 1; }
  if (data_directory.empty()) {
    // Get the input data environment variable
    auto input_path = std::getenv("HOLOSCAN_INPUT_PATH");
    if (input_path == nullptr || input_path[0] == '\0') {
      HOLOSCAN_LOG_ERROR(
          "Input data not provided. Use --data or set HOLOSCAN_INPUT_PATH environment variable.");
      exit(-1);
    }
    data_directory = std::string(input_path);
    HOLOSCAN_LOG_INFO("Using input data from {}", data_directory);
  }

  if (config_path.empty()) {
    // Get the input data environment variable
    auto config_file_path = std::getenv("HOLOSCAN_CONFIG_PATH");
    if (config_file_path == nullptr || config_file_path[0] == '\0') {
      auto config_file = std::filesystem::canonical(argv[0]).parent_path();
      config_path = config_file / std::filesystem::path("endoscopy_tool_tracking.yaml");
    } else {
      HOLOSCAN_LOG_INFO("Using configuration file from {}", data_directory);
      config_path = config_file_path;
    }
  }

  auto app = holoscan::make_application<App>();
  app->config(config_path);
  app->set_datapath(data_directory);

  auto trackers = app->track_distributed();
  app->run();
  for (const auto& [name, tracker] : trackers) {
    std::cout << "Fragment: " << name << std::endl;
    tracker->print();
  }
  return 0;
}
