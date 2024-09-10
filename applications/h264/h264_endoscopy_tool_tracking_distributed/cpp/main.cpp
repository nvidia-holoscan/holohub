/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
#include <iostream>
#include <string>

#include <holoscan/holoscan.hpp>

#include "cloud_inference_fragment.hpp"
#include "video_input_fragment.hpp"
#include "viz_fragment.hpp"

using namespace holoscan;

class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath_ = path; }

  void compose() override {
    using namespace holoscan;

    auto width = 854;
    auto height = 480;

    auto video_in = make_fragment<VideoInputFragment>("video_in", datapath_);
    auto video_in_fragment = std::dynamic_pointer_cast<VideoInputFragment>(video_in);
    auto cloud_inference =
        make_fragment<CloudInferenceFragment>("inference", datapath_, width, height);
    auto viz = make_fragment<VizFragment>("viz", width, height);

    add_flow(video_in,
             cloud_inference,
             {{"bitstream_reader.output_transmitter", "video_decoder_request.input_frame"}});
    add_flow(video_in, viz, {{"decoder_output_format_converter.tensor", "holoviz.receivers"}});
    add_flow(cloud_inference,
             viz,
             {{"tool_tracking_postprocessor.out_coords", "holoviz.receivers"},
              {"tool_tracking_postprocessor.out_mask", "holoviz.receivers"}});
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
  // Parse the arguments
  std::string config_path = "";
  std::string data_directory = "";
  if (!parse_arguments(argc, argv, data_directory, config_path)) { return 1; }
  if (data_directory.empty()) {
    // Get the input data environment variable
    auto input_path = std::getenv("HOLOSCAN_INPUT_PATH");
    if (input_path != nullptr && input_path[0] != '\0') {
      data_directory = std::string(input_path);
    } else if (std::filesystem::is_directory(std::filesystem::current_path() / "data/endoscopy")) {
      data_directory = std::string((std::filesystem::current_path() / "data/endoscopy").c_str());
    } else {
      HOLOSCAN_LOG_ERROR(
          "Input data not provided. Use --data or set HOLOSCAN_INPUT_PATH environment variable.");
      exit(-1);
    }
  }

  if (config_path.empty()) {
    // Get the input data environment variable
    auto config_file_path = std::getenv("HOLOSCAN_CONFIG_PATH");
    if (config_file_path == nullptr || config_file_path[0] == '\0') {
      auto config_file = std::filesystem::canonical(argv[0]).parent_path();
      config_path = config_file / std::filesystem::path("endoscopy_tool_tracking.yaml");
    } else {
      config_path = config_file_path;
    }
  }

  auto app = holoscan::make_application<App>();

  HOLOSCAN_LOG_INFO("Using configuration file from {}", config_path);
  app->config(config_path);

  HOLOSCAN_LOG_INFO("Using input data from {}", data_directory);
  app->set_datapath(data_directory);

  app->run();

  return 0;
}
