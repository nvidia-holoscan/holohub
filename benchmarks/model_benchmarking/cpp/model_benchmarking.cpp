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
#include <memory>

#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include "holoscan/holoscan.hpp"

namespace holoscan::ops {

/**
 * @brief This Operator takes an input tensor and does nothing with it. It works as a sink
 * without any function.
 *
 */
class SinkOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SinkOp)

  SinkOp() = default;

  void setup(OperatorSpec& spec) { spec.input<std::any>("in"); }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    auto value = op_input.receive<std::any>("in");
  }
};
}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  App() = default;
  App(const std::string& datapath, const std::string& model_name, int num_inferences,
      bool only_inference, bool inference_postprocessing, bool headless, bool replayer)
      : datapath(datapath),
        model_name(model_name),
        num_inferences(num_inferences),
        only_inference(only_inference),
        inference_postprocessing(inference_postprocessing),
        headless(headless),
        replayer(replayer) {
    holoscan::Application();
    if (!std::filesystem::exists(datapath)) {
      std::cerr << "Data path " << datapath << " does not exist." << std::endl;
      exit(1);
    }

    std::string model_path =
        datapath.back() == '/' ? (datapath + model_name) : (datapath + "/" + model_name);
    if (!std::filesystem::exists(model_path)) {
      std::cerr << "Model path " << model_path << " does not exist." << std::endl;
      exit(1);
    }
  }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Resource> pool_resource = make_resource<UnboundedAllocator>("pool");
    std::shared_ptr<Operator> source;

    if (replayer) {
      source = make_operator<ops::VideoStreamReplayerOp>(
          "replayer", from_config("replayer"), Arg("directory") = datapath);

    } else {
      source = make_operator<ops::V4L2VideoCaptureOp>(
          "source", from_config("source"), Arg("allocator") = pool_resource);
    }

    auto preprocessor = make_operator<ops::FormatConverterOp>(
        "preprocessor", from_config("preprocessor"), Arg("pool") = pool_resource);
    ops::InferenceOp::DataMap model_path_map;
    ops::InferenceOp::DataVecMap pre_processor_map;
    ops::InferenceOp::DataVecMap inference_map;

    for (int i = 0; i < num_inferences; i++) {
      std::string model_index_str = "own_model_" + std::to_string(i);

      model_path_map.insert(model_index_str, datapath + "/" + model_name);
      pre_processor_map.insert(model_index_str, {"source_video"});

      std::string output_name = "output" + std::to_string(i);
      inference_map.insert(model_index_str, {output_name});
    }
    auto inference = make_operator<ops::InferenceOp>("inference",
                                                     from_config("inference"),
                                                     Arg("allocator") = pool_resource,
                                                     Arg("model_path_map", model_path_map),
                                                     Arg("pre_processor_map", pre_processor_map),
                                                     Arg("inference_map", inference_map));

    std::vector<std::shared_ptr<Operator>> holovizs;
    holovizs.reserve(num_inferences);

    if (!only_inference && !inference_postprocessing) {
      for (int i = 0; i < num_inferences; i++) {
        std::string holoviz_name = "holoviz" + std::to_string(i);
        auto holoviz = make_operator<ops::HolovizOp, std::string>(
            holoviz_name, from_config("viz"), Arg("headless") = headless);
        holovizs.push_back(holoviz);
        // Passthrough to Visualization
        if (replayer) {
          add_flow(source, holoviz, {{"", "receivers"}});
        } else {
          add_flow(source, holoviz, {{"signal", "receivers"}});
        }
      }
    }

    // Inference Path
    if (replayer) {
      add_flow(source, preprocessor, {{"", "source_video"}});
    } else {
      add_flow(source, preprocessor, {{"signal", "source_video"}});
    }
    add_flow(preprocessor, inference, {{"tensor", "receivers"}});
    if (only_inference) {
      HOLOSCAN_LOG_INFO(
          "Only inference mode is on, no post-processing and visualization will be done.");
      auto sink = make_operator<ops::SinkOp>("sink");
      add_flow(inference, sink);
      return;
    }

    std::vector<std::shared_ptr<Operator>> postprocessors;
    postprocessors.reserve(num_inferences);

    for (int i = 0; i < num_inferences; i++) {
      std::string postprocessor_name = "postprocessor" + std::to_string(i);
      std::string in_tensor_name = "output" + std::to_string(i);
      auto postprocessor = make_operator<ops::SegmentationPostprocessorOp, std::string>(
          postprocessor_name,
          from_config("postprocessor"),
          Arg("allocator") = pool_resource,
          Arg("in_tensor_name") = in_tensor_name);
      postprocessors.push_back(postprocessor);
      add_flow(inference, postprocessor, {{"transmitter", "in_tensor"}});
    }

    if (inference_postprocessing) {
      HOLOSCAN_LOG_INFO("Inference and Post-processing mode is on. No visualization will be done.");
      for (int i = 0; i < num_inferences; i++) {
        std::string sink_name = "sink" + std::to_string(i);
        auto sink = make_operator<ops::SinkOp, std::string>(sink_name);
        add_flow(postprocessors[i], sink);
      }
      return;
    }

    for (int i = 0; i < num_inferences; i++) {
      add_flow(postprocessors[i], holovizs[i], {{"out_tensor", "receivers"}});
    }
  }

 private:
  int num_inferences = 1;
  bool only_inference = false, inference_postprocessing = false, headless = false, replayer = false;
  std::string datapath, model_name;
};

void print_help() {
  std::cout << "Usage: model_benchmarking [OPTIONS] [ConfigPath]" << std::endl;
  std::cout << "ConfigPath                    Path to the config file (default: "
               "<current directory>/model_benchmarking.yaml)"
            << std::endl;
  std::cout << "Options:" << std::endl;
  std::cout << "  -d, --data <path>               Path to the data directory (default: "
               "/workspace/holohub/data/ultrasound_segmentation)"
            << std::endl;
  std::cout
      << "  -m, --model-name <path>              name of the model file with extension (default: "
         "us_unet_256x256_nhwc.onnx)"
      << std::endl;
  std::cout << "  -i, --only-inference            Only run inference, no post-processing or "
               "visualization"
            << std::endl;
  std::cout << "  -p, --inference-postprocessing  Run inference and post-processing, no "
               "visualization"
            << std::endl;
  std::cout
      << "  -l, --multi-inference <num>     Number of inferences to run in parallel (default: 1)"
      << std::endl;
  std::cout << "  -e, --headless                  Run holoviz in headless mode." << std::endl;
  std::cout << "  -h, --help                      Print this help" << std::endl;
}

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path,
                     std::string& model_name, bool& only_inference, bool& inference_postprocessing,
                     int& num_inferences, bool& headless, bool& replayer) {
  static struct option long_options[] = {{"help", required_argument, 0, 'h'},
                                         {"data", required_argument, 0, 'd'},
                                         {"model-name", required_argument, 0, 'm'},
                                         {"only-inference", optional_argument, 0, 'i'},
                                         {"inference-postprocessing", optional_argument, 0, 'p'},
                                         {"headless", optional_argument, 0, 'e'},
                                         {"replayer", optional_argument, 0, 'r'},
                                         {"multi-inference", required_argument, 0, 'l'},
                                         {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "hd:m:v:iperl:", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'h':
        print_help();
        return false;
      case 'd':
        data_path = optarg;
        break;
      case 'm':
        model_name = optarg;
        break;
      case 'i':
        only_inference = true;
        break;
      case 'p':
        inference_postprocessing = true;
        break;
      case 'e':
        headless = true;
        break;
      case 'r':
        replayer = true;
        break;
      case 'l':
        num_inferences = std::stoi(optarg);
        break;
      default:
        std::cerr << "Unknown arguments returned: " << c << std::endl;
        print_help();
        return false;
    }
  }

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}

int main(int argc, char** argv) {
  // Parse the arguments
  std::string config_name = "";
  std::string data_path = "/workspace/holohub/data/ultrasound_segmentation";
  std::string model_name = "us_unet_256x256_nhwc.onnx";
  bool only_inference = false, inference_postprocessing = false, headless = false, replayer = false;
  int num_inferences = 1;
  if (!parse_arguments(argc,
                       argv,
                       config_name,
                       data_path,
                       model_name,
                       only_inference,
                       inference_postprocessing,
                       num_inferences,
                       headless,
                       replayer)) {
    return 1;
  }

  auto app = holoscan::make_application<App>(data_path,
                                             model_name,
                                             num_inferences,
                                             only_inference,
                                             inference_postprocessing,
                                             headless,
                                             replayer);
  if (config_name != "") {
    // Check if config_name is a valid path
    if (!std::filesystem::exists(config_name)) {
      std::cerr << "Config file " << config_name << " does not exist." << std::endl;
      return 0;
    }
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/model_benchmarking.yaml";
    app->config(config_path);
  }

  app->run();

  return 0;
}
