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

#include <getopt.h>

#include <holoscan/holoscan.hpp>

// Holoscan SDK Operators
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/inference/inference.hpp>
// Holohub operators
#include <videomaster_source.hpp>
// Orsi: Holoscan native operators
#include <format_converter.hpp>
#include <segmentation_postprocessor.hpp>
#include <segmentation_preprocessor.hpp>
#include <orsi_visualizer.hpp>


enum class VideoSource { REPLAYER, VIDEOMASTER };

class App : public holoscan::Application {
private:
  VideoSource video_source_ = VideoSource::REPLAYER;
  std::string datapath = "data";

 public:
  void set_source(const std::string& source) {
    if (source == "videomaster") { video_source_ = VideoSource::VIDEOMASTER; }
    if (source == "replayer") { video_source_ = VideoSource::REPLAYER; }
  }

  void set_datapath(const std::string& path) {
     datapath = path;
  }


  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;
    std::shared_ptr<Operator> drop_alpha_channel;
    std::shared_ptr<Resource> allocator_resource = make_resource<UnboundedAllocator>("unbounded_allocator");


    switch (video_source_) {
      case VideoSource::VIDEOMASTER:
        source = make_operator<ops::VideoMasterSourceOp>(
            "videomaster",
            from_config("videomaster"),
            Arg("pool") = allocator_resource);
        break;
      default:
        source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"), 
                                                            Arg("directory", datapath + "/video"));
        break;
    }

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
    make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    const int width = 1920;
    const int height = 1080;
    const int n_channels = 4;
    const int bpp = 4;

    // -------------------------------------------------------------------------------------
    //
    // Format conversion operators
    //

    if (video_source_ == VideoSource::VIDEOMASTER) {
      uint64_t drop_alpha_block_size = width * height * n_channels * bpp;
      uint64_t drop_alpha_num_blocks = 2;
      drop_alpha_channel = make_operator<ops::orsi::FormatConverterOp>(
          "drop_alpha_channel",
          from_config("drop_alpha_channel_videomaster"),
           Arg("allocator") = make_resource<BlockMemoryPool>(
              "pool", 1, drop_alpha_block_size, drop_alpha_num_blocks),
          Arg("cuda_stream_pool") = cuda_stream_pool);
    }

    int width_preprocessor = 1264;
    int height_preprocessor = 1080;
    uint64_t preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp;
    uint64_t preprocessor_num_blocks = 2;
    auto format_converter = make_operator<ops::orsi::FormatConverterOp>(
        "format_converter",
        from_config("format_converter"),
        Arg("in_tensor_name",
            std::string(video_source_ == VideoSource::VIDEOMASTER ? "source_video" : "")),
            Arg("allocator") = allocator_resource);

    // -------------------------------------------------------------------------------------
    //
    // Pre-processing Operators
    //

    auto segmentation_preprocessor = make_operator<ops::orsi::SegmentationPreprocessorOp>(
        "segmentation_preprocessor",
        from_config("segmentation_preprocessor"),
        Arg("allocator") = allocator_resource);

    // -------------------------------------------------------------------------------------
    //
    // Multi-AI Inference Operator
    //

    ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("tool_segmentation", datapath + "/model/fpn_one_output_nhwc.onnx");

    auto multiai_inference = make_operator<ops::InferenceOp>(
      "multiai_inference", from_config("multiai_inference"),
      Arg("model_path_map", model_path_map),
      Arg("allocator") = allocator_resource
    );

    // -------------------------------------------------------------------------------------
    //
    // Post-processing Operators
    //

    auto segmentation_postprocessor = make_operator<ops::orsi::SegmentationPostprocessorOp>(
        "segmentation_postprocessor",
        from_config("segmentation_postprocessor"),
        Arg("allocator") = allocator_resource
        );

    // -------------------------------------------------------------------------------------
    //
    // Visualization Operator
    //

    auto segmentation_visualizer =
        make_operator<ops::orsi::OrsiVisualizationOp>("segmentation_visualizer",
                                      from_config("segmentation_visualizer"),
                                      Arg("stl_file_path" , datapath + "/stl/stent_example_case/"),
                                      Arg("allocator") = allocator_resource);

    // Flow definition
    switch (video_source_) {
      case VideoSource::VIDEOMASTER:
        add_flow(source, segmentation_visualizer, {{"signal", "receivers"}});
        add_flow(source, drop_alpha_channel, {{"signal", ""}});
        add_flow(drop_alpha_channel, format_converter);
        break;
      case VideoSource::REPLAYER:
      default:
        add_flow(source, segmentation_visualizer, {{"", "receivers"}});
        add_flow(source, format_converter);
        break;
    }

    add_flow(format_converter, segmentation_preprocessor);
    add_flow(segmentation_preprocessor, multiai_inference, {{"", "receivers"}});
    add_flow(multiai_inference, segmentation_postprocessor, {{"transmitter", ""}});
    add_flow(segmentation_postprocessor, segmentation_visualizer, {{"", "receivers"}});
  }

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
    config_path += "/app_config.yaml";
    app->config(config_path);
  }

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);
  if (data_path != "") app->set_datapath(data_path);
   auto& tracker = app->track(); 
  app->run();
  std::cout << "// Application::run completed. Printing tracker results" << std::endl;
  tracker.print();

  return 0;
}
