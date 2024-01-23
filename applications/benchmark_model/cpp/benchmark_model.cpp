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
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include "holoscan/holoscan.hpp"

bool only_inference = false;
bool inference_postprocessing = false;
int num_inferences = 1;

namespace holoscan::ops {

/**
 * @brief This Operator takes an input tensor and does nothing with it. It works as a sink
 * without any function.
 *
 */
class NoOpSink : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(NoOpSink)

  NoOpSink() = default;

  void setup(OperatorSpec& spec) { spec.input<std::any>("in"); }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    auto value = op_input.receive<std::any>("in");
  }
};
}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  App() = default;
  App(const std::string& videopath, const std::string& modelpath)
      : videopath(videopath), modelpath(modelpath) {
    holoscan::Application();
  }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;
    std::shared_ptr<Resource> pool_resource = make_resource<UnboundedAllocator>("pool");
    source = make_operator<ops::VideoStreamReplayerOp>(
        "replayer", from_config("replayer"), Arg("directory", videopath));

    auto preprocessor = make_operator<ops::FormatConverterOp>(
        "preprocessor", from_config("preprocessor"), Arg("pool") = pool_resource);

    std::cout << "Model Path is " << modelpath << std::endl;
    std::cout << "Video Path is " << videopath << std::endl;

    ops::InferenceOp::DataMap model_path_map;
    ops::InferenceOp::DataVecMap pre_processor_map;
    ops::InferenceOp::DataVecMap inference_map;
    for (int i = 0; i < num_inferences; i++) {
      model_path_map.insert("own_model_" + std::to_string(i),
                            modelpath + "/us_unet_256x256_nhwc.onnx");
      pre_processor_map.insert("own_model_" + std::to_string(i), {"source_video"});
      inference_map.insert("own_model_" + std::to_string(i), {"output" + std::to_string(i)});
    }
    auto inference = make_operator<ops::InferenceOp>("inference",
                                                     from_config("inference"),
                                                     Arg("allocator") = pool_resource,
                                                     Arg("model_path_map", model_path_map),
                                                     Arg("pre_processor_map", pre_processor_map),
                                                     Arg("inference_map", inference_map));

    auto holoviz =
        make_operator<ops::HolovizOp>("viz", from_config("viz"), Arg("allocator") = pool_resource);

    // Flow definition

    // Passthrough to Visualization
    if (!only_inference && !inference_postprocessing) {
      add_flow(source, holoviz, {{"output", "receivers"}});
    }

    // Inference Path
    add_flow(source, preprocessor, {{"output", "source_video"}});
    add_flow(preprocessor, inference, {{"tensor", "receivers"}});
    if (only_inference) {
      HOLOSCAN_LOG_INFO(
          "Only inference mode is on, no post-processing and visualization will be done.");
      auto sink = make_operator<ops::NoOpSink>("sink");
      add_flow(inference, sink);
      return;
    }
    auto postprocessor = make_operator<ops::SegmentationPostprocessorOp>(
        "postprocessor", from_config("postprocessor"), Arg("allocator") = pool_resource);
    add_flow(inference, postprocessor, {{"transmitter", "in_tensor"}});

    if (inference_postprocessing) {
      HOLOSCAN_LOG_INFO("Inference and Post-processing mode is on. No visualization will be done.");
      auto sink = make_operator<ops::NoOpSink>("sink");
      add_flow(postprocessor, sink);
      return;
    }

    add_flow(postprocessor, holoviz, {{"out_tensor", "receivers"}});
  }

 private:
  bool is_aja_source_ = false;
  std::string videopath = "../data";
  std::string modelpath = "model/";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path,
                     bool& only_inference, bool& inference_postprocessing, int& num_inferences) {
  static struct option long_options[] = {{"data", required_argument, 0, 'd'},
                                         {"only-inference", optional_argument, 0, 'i'},
                                         {"inference-postprocessing", optional_argument, 0, 'p'},
                                         {"multi-inference", required_argument, 0, 'm'},
                                         {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "dipm", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      case 'i':
        only_inference = true;
        break;
      case 'p':
        inference_postprocessing = true;
        break;
      case 'm':
        num_inferences = std::stoi(optarg);
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}

int main(int argc, char** argv) {
  // Parse the arguments
  std::string config_name = "";
  // get std::string from std::getenv("HOLOSCAN_INPUT_PATH")
  auto env_var = std::getenv("HOLOSCAN_INPUT_PATH");
  auto data_path = std::string((env_var == nullptr ? "" : env_var)) + "../data";
  if (!parse_arguments(argc,
                       argv,
                       config_name,
                       data_path,
                       only_inference,
                       inference_postprocessing,
                       num_inferences)) {
    return 1;
  }

  auto app = holoscan::make_application<App>(data_path, data_path);
  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/byom.yaml";
    app->config(config_path);
  }

  app->run();

  return 0;
}