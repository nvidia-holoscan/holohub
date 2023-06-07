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
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <qcap_source.hpp>

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) {
    if (source == "aja") { is_aja_source_ = true; }
    if (source == "qcap") { is_qcap_source_ = true; }
  }

  void set_datapath(const std::string& path) {
     datapath = path;
  }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;
    std::shared_ptr<Operator> drop_alpha_channel;

    if (is_aja_source_) {
      source = make_operator<ops::AJASourceOp>("aja", from_config("aja"));
    } else if (is_qcap_source_) {
      source = make_operator<ops::QCAPSourceOp>("qcap", from_config("qcap"));
    } else {
      source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"),
                                                                     Arg("directory", datapath));
    }

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    const int width = 1920;
    const int height = 1080;
    const int n_channels = 4;
    const int bpp = 4;
    if (is_aja_source_) {
      uint64_t drop_alpha_block_size = width * height * n_channels * bpp;
      uint64_t drop_alpha_num_blocks = 2;
      drop_alpha_channel = make_operator<ops::FormatConverterOp>(
          "drop_alpha_channel",
          from_config("drop_alpha_channel"),
          Arg("pool") = make_resource<BlockMemoryPool>(
              "pool", 1, drop_alpha_block_size, drop_alpha_num_blocks),
          Arg("cuda_stream_pool") = cuda_stream_pool);
    }

    int width_preprocessor = 1264;
    int height_preprocessor = 1080;
    uint64_t preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp;
    uint64_t preprocessor_num_blocks = 2;
    auto segmentation_preprocessor = make_operator<ops::FormatConverterOp>(
        "segmentation_preprocessor",
        from_config("segmentation_preprocessor"),
        Arg("in_tensor_name", std::string(is_aja_source_ ? "source_video" : "")),
        Arg("pool") = make_resource<BlockMemoryPool>(
            "pool", 1, preprocessor_block_size, preprocessor_num_blocks),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    const int n_channels_inference = 2;
    const int width_inference = 256;
    const int height_inference = 256;
    const int bpp_inference = 4;
    const uint64_t inference_block_size =
        width_inference * height_inference * n_channels_inference * bpp_inference;
    const uint64_t inference_num_blocks = 2;

    ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("ultrasound_seg", datapath + "/us_unet_256x256_nhwc.onnx");
    auto segmentation_inference = make_operator<ops::InferenceOp>(
        "segmentation_inference_holoinfer",
        from_config("segmentation_inference_holoinfer"),
        Arg("model_path_map", model_path_map),
        Arg("allocator") =
            make_resource<BlockMemoryPool>("pool", 1, inference_block_size, inference_num_blocks));

    const uint64_t postprocessor_block_size = width_inference * height_inference;
    const uint64_t postprocessor_num_blocks = 2;
    auto segmentation_postprocessor = make_operator<ops::SegmentationPostprocessorOp>(
        "segmentation_postprocessor",
        from_config("segmentation_postprocessor"),
        Arg("allocator") = make_resource<BlockMemoryPool>(
            "pool", 1, postprocessor_block_size, postprocessor_num_blocks));

    auto segmentation_visualizer =
        make_operator<ops::HolovizOp>("segmentation_visualizer",
                                      from_config("segmentation_visualizer"),
                                      Arg("allocator") = make_resource<UnboundedAllocator>("pool"),
                                      Arg("cuda_stream_pool") = cuda_stream_pool);

    // Flow definition

    if (is_aja_source_) {
      add_flow(source, segmentation_visualizer, {{"video_buffer_output", "receivers"}});
      add_flow(source, drop_alpha_channel, {{"video_buffer_output", ""}});
      add_flow(drop_alpha_channel, segmentation_preprocessor);
    } else {
      add_flow(source, segmentation_visualizer, {{"", "receivers"}});
      add_flow(source, segmentation_preprocessor);
    }

    add_flow(segmentation_preprocessor, segmentation_inference, {{"", "receivers"}});
    add_flow(segmentation_inference, segmentation_postprocessor, {{"transmitter", ""}});
    add_flow(segmentation_postprocessor, segmentation_visualizer, {{"", "receivers"}});
  }

 private:
  bool is_aja_source_ = false;
  bool is_qcap_source_ = false;
  std::string datapath = "data/ultrasound_segmentation";
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
    config_path += "/ultrasound_segmentation.yaml";
    app->config(config_path);
  }

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);
  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
