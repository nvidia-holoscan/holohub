/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/inference_processor/inference_processor.hpp>

#ifdef AJA_SOURCE
#include <aja_source.hpp>
#endif

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) {
    if (source == "aja") { is_aja_source_ = true; }
  }

  void set_datapath(const std::string& path) { datapath = path; }

  void compose() override {
    using namespace holoscan;
    const bool enable_analytics = from_config("enable_analytics").as<bool>();

    std::shared_ptr<Operator> source;
    std::shared_ptr<Resource> pool_resource = make_resource<UnboundedAllocator>("pool");
    if (is_aja_source_) {
#ifdef AJA_SOURCE
      source = make_operator<ops::AJASourceOp>("aja", from_config("aja"));
#else
      throw std::runtime_error(
          "AJA is requested but not available. Please enable AJA at build time.");
#endif
    } else {
      const std::string replayer_config = enable_analytics ? "analytics_replayer" : "replayer";
      source = make_operator<ops::VideoStreamReplayerOp>(
          "replayer", from_config(replayer_config), Arg("directory", datapath));
    }

    auto in_dtype = is_aja_source_ ? std::string("rgba8888") : std::string("rgb888");

    auto out_of_body_preprocessor =
        make_operator<ops::FormatConverterOp>("out_of_body_preprocessor",
                                              from_config("out_of_body_preprocessor"),
                                              Arg("in_dtype") = in_dtype,
                                              Arg("pool") = pool_resource);

    ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("out_of_body", datapath + "/out_of_body_detection.onnx");

    auto out_of_body_inference =
        make_operator<ops::InferenceOp>("out_of_body_inference",
                                        from_config("out_of_body_inference"),
                                        Arg("model_path_map", model_path_map),
                                        Arg("allocator") = pool_resource);

    const std::string out_of_body_postprocessor_config =
        enable_analytics ? "analytics_out_of_body_postprocessor" : "out_of_body_postprocessor";
    auto out_of_body_postprocessor =
        make_operator<ops::InferenceProcessorOp>("out_of_body_postprocessor",
                                                 from_config(out_of_body_postprocessor_config),
                                                 Arg("allocator") = pool_resource,
                                                 Arg("disable_transmitter") = true);

    // Flow definition
    if (is_aja_source_) {
#ifdef AJA_SOURCE
      const std::set<std::pair<std::string, std::string>> aja_ports = {{"video_buffer_output", ""}};
      add_flow(source, out_of_body_preprocessor, aja_ports);
#else
      throw std::runtime_error(
          "AJA is requested but not available. Please enable AJA at build time.");
#endif
    } else {
      add_flow(source, out_of_body_preprocessor);
    }

    add_flow(out_of_body_preprocessor, out_of_body_inference, {{"", "receivers"}});
    add_flow(out_of_body_inference, out_of_body_postprocessor, {{"transmitter", "receivers"}});
  }

 private:
  bool is_aja_source_ = false;
  std::string datapath = "data/endoscopy_out_of_body_detection";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {
    {"config", required_argument, 0, 'c'},
    {"data", required_argument, 0, 'd'},
    {0, 0, 0, 0}
  };

  while (int c = getopt_long(argc, argv, "c:d:", long_options, NULL)) {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'c':
        config_name = optarg;
        break;
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  return true;
}

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return 1; }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/endoscopy_out_of_body_detection.yaml";
    app->config(config_path);
  }

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);
  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
