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
#include <csignal>
#include <iostream>
#include <string>
#include <thread>

#include <grpc_server.hpp>
#include <holoscan/holoscan.hpp>

#include "app_cloud_pipeline.hpp"
#include "grpc_service.hpp"

using namespace holoscan;
using namespace holohub::grpc_h264_endoscopy_tool_tracking;

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, uint32_t& port, std::string& data_path,
                     std::string& config_path) {
  static struct option long_options[] = {{"port", required_argument, 0, 'p'},
                                         {"data", required_argument, 0, 'd'},
                                         {"config", required_argument, 0, 'c'},
                                         {0, 0, 0, 0}};

  int c;
  while (optind < argc) {
    if ((c = getopt_long(argc, argv, "p:", long_options, NULL)) != -1) {
      switch (c) {
        case 'c':
          config_path = optarg;
          break;
        case 'd':
          data_path = optarg;
          break;
        case 'p':
          try {
            port = std::stoi(optarg);
            if (port < 0 || port > 65535) { throw std::out_of_range("port number out of range"); }
          } catch (const std::exception& e) { std::cerr << e.what() << ":" << optarg << '\n'; }
          break;
        default:
          HOLOSCAN_LOG_ERROR("Unhandled option '{}'", static_cast<char>(c));
          return false;
      }
    }
  }

  return true;
}

void signal_handler(int signum) {
  HOLOSCAN_LOG_WARN("Caught signal {}. Stopping services...", signum);
  std::thread myThread([] { GrpcService::get_instance(0, nullptr).stop(); });
  myThread.join();
}

/** Helper function to parse benchmarking setting from the configuration file */
void parse_config(const std::string& config_path, bool& benchmarking) {
  auto config = holoscan::Config(config_path);
  auto& yaml_nodes = config.yaml_nodes();
  for (const auto& yaml_node : yaml_nodes) {
    try {
      auto application = yaml_node["application"];
      if (application.IsMap()) { benchmarking = application["benchmarking"].as<bool>(); }
    } catch (std::exception& e) {
      HOLOSCAN_LOG_ERROR("Error parsing configuration file: {}", e.what());
      benchmarking = false;
    }
  }
}

/** Main function */
/**
 * @file app_cloud_main.cpp
 * @brief Main entry point for the gRPC H264 Endoscopy Tool Tracking application.
 *
 * This application sets up and starts a gRPC server for endoscopy tool tracking using H264 video
 * streams.
 *
 * The main function performs the following steps:
 * 1. Parses command-line arguments to get the port number, data directory, and configuration file
 * path.
 * 2. If the configuration file path is not provided, it attempts to retrieve it from the
 * environment variable `HOLOSCAN_CONFIG_PATH` or defaults to a file named
 * `endoscopy_tool_tracking.yaml` in the executable's directory.
 * 3. If the data directory is not provided, it attempts to retrieve it from the environment
 * variable `HOLOSCAN_INPUT_PATH` or defaults to a directory named `data/endoscopy` in the current
 * working directory.
 * 4. Registers the Endoscopy Tool Tracking application with the `ApplicationFactory`.
 * 5. Starts the gRPC service on the specified port.
 *
 * @param argc The number of command-line arguments.
 * @param argv The array of command-line arguments.
 * @return Returns 0 on successful execution, or 1 if argument parsing fails.
 */
int main(int argc, char** argv) {
  // Parse the arguments
  uint32_t port = 50051;
  std::string config_path = "";
  std::string data_directory = "";

  if (!parse_arguments(argc, argv, port, data_directory, config_path)) { return 1; }

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

  bool benchmarking = false;
  parse_config(config_path, benchmarking);

  // Register each gRPC service with a Holoscan application:
  // - the callback function (create_application_instance_func) is used to create a new instance of
  //   the application when a new RPC call is received.
  ApplicationFactory::get_instance()->register_application(
      "EntityStream",
      [config_path, data_directory, benchmarking](
          std::queue<std::shared_ptr<nvidia::gxf::Entity>> incoming_request_queue,
          std::queue<std::shared_ptr<EntityResponse>>
              outgoing_response_queue) {
        ApplicationInstance application_instance;
        application_instance.instance = holoscan::make_application<AppCloudPipeline>(
            incoming_request_queue, outgoing_response_queue);
        application_instance.instance->config(config_path);
        application_instance.instance->set_data_path(data_directory);
        if (benchmarking) {
          application_instance.tracker = &application_instance.instance->track();
        }
        application_instance.future = application_instance.instance->run_async();
        return application_instance;
      });

  struct sigaction sa;
  memset(&sa, 0, sizeof(sa));
  sa.sa_handler = signal_handler;
  sigaction(SIGINT, &sa, NULL);
  sigaction(SIGHUP, &sa, NULL);
  sigaction(SIGTERM, &sa, NULL);

  try {
    GrpcService::get_instance(port, ApplicationFactory::get_instance()).start();
  } catch (std::exception& e) {
    HOLOSCAN_LOG_ERROR("Error running gRPC service: {}", e.what());
    exit(-1);
  } catch (...) {
    HOLOSCAN_LOG_ERROR("Unknown error running gRPC service");
    exit(-2);
  }
  return 0;
}
