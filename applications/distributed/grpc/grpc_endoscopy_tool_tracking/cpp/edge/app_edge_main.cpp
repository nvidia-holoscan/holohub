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

// Run th edge app with a single fragment
#include "app_edge_single_fragment.hpp"

// Run the edge app with two fragments
#include "app_edge_multi_fragment.hpp"

using namespace holoscan;
using namespace holohub::grpc_h264_endoscopy_tool_tracking;

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& data_path, std::string& config_path) {
  static struct option long_options[] = {
      {"data", required_argument, 0, 'd'}, {"config", required_argument, 0, 'c'}, {0, 0, 0, 0}};

  int c;
  while (optind < argc) {
    if ((c = getopt_long(argc, argv, "d:c:", long_options, NULL)) != -1) {
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
  }

  return true;
}

/** Helper function to parse fragment mode and benchmarking settings from the configuration file */
void parse_config(const std::string& config_path, bool& multi_fragment_mode, bool& benchmarking) {
  auto config = holoscan::Config(config_path);
  auto& yaml_nodes = config.yaml_nodes();
  for (const auto& yaml_node : yaml_nodes) {
    try {
      auto application = yaml_node["application"];
      if (application.IsMap()) {
        multi_fragment_mode = application["multifragment"].as<bool>();
        benchmarking = application["benchmarking"].as<bool>();
      }
    } catch (std::exception& e) {
      HOLOSCAN_LOG_ERROR("Error parsing configuration file: {}", e.what());
      multi_fragment_mode = false;
      benchmarking = false;
    }
  }
}

/** Main function */
/**
 * @file app_edge_main.cpp
 * @brief Main entry point for the edge (client) side of the H.264 endoscopy tool tracking
 * application.
 *
 * This file contains the main function which initializes and runs the application.
 * It handles argument parsing, configuration, and data directory setup.
 *
 * @param argc Number of command-line arguments.
 * @param argv Array of command-line arguments.
 * @return int Exit status of the application.
 *
 * The main function performs the following steps:
 * 1. Parses command-line arguments to obtain the data directory and configuration path.
 * 2. If the data directory is not provided, it attempts to retrieve it from the environment
 * variable `HOLOSCAN_INPUT_PATH` or defaults to a local "data/endoscopy" directory.
 * 3. If the configuration path is not provided, it attempts to retrieve it from the environment
 * variable `HOLOSCAN_CONFIG_PATH` or defaults to a local "endoscopy_tool_tracking.yaml" file.
 * 4. Creates an instance of the application (`AppEdge`).
 * 5. Configures the application with the provided configuration file.
 * 6. Sets the data path for the application.
 * 7. Runs the application.
 */
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

  bool multi_fragment_mode = false;
  bool benchmarking = false;
  parse_config(config_path, multi_fragment_mode, benchmarking);
  if (multi_fragment_mode) {
    HOLOSCAN_LOG_INFO("Running application in multi-fragment mode");
    auto app = holoscan::make_application<AppEdgeMultiFragment>();

    HOLOSCAN_LOG_INFO("Using configuration file from {}", config_path);
    app->config(config_path);

    HOLOSCAN_LOG_INFO("Using input data from {}", data_directory);
    app->set_datapath(data_directory);

    std::unordered_map<std::string, DataFlowTracker*> trackers;
    if (benchmarking) {
      HOLOSCAN_LOG_INFO("Benchmarking enabled");
      trackers = app->track_distributed();
    }

    app->run();

    if (benchmarking) {
      for (const auto& [name, tracker] : trackers) {
        std::cout << "Fragment: " << name << std::endl;
        tracker->print();
      }
    }
  } else {
    HOLOSCAN_LOG_INFO("Running application in single fragment mode");
    auto app = holoscan::make_application<AppEdgeSingleFragment>();

    HOLOSCAN_LOG_INFO("Using configuration file from {}", config_path);
    app->config(config_path);

    HOLOSCAN_LOG_INFO("Using input data from {}", data_directory);
    app->set_datapath(data_directory);

    DataFlowTracker* tracker = nullptr;
    if (benchmarking) {
      HOLOSCAN_LOG_INFO("Benchmarking enabled");
      tracker = &app->track();
    }
    app->run();
    if (benchmarking) { tracker->print(); }
  }
  return 0;
}
