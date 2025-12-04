/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <filesystem>
#include <holoscan/holoscan.hpp>
#include <string>

#include "publisher.h"
#include "subscriber.h"

/** Helper function to parse command line arguments */
bool parse_arguments(int argc, char** argv, std::string& data_path, std::string& config_path,
                     std::string& hostname, int& port, std::string& mode) {
  static struct option long_options[] = {
      {"data", required_argument, 0, 'd'},
      {"config", required_argument, 0, 'c'},
      {"hostname", required_argument, 0, 'h'},
      {"port", required_argument, 0, 'p'},
      {"mode", required_argument, 0, 'm'},
      {"help", no_argument, 0, '?'},
      {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d:c:h:p:m:?", long_options, NULL)) {
    if (c == -1) break;

    switch (c) {
      case 'c':
        config_path = optarg;
        break;
      case 'd':
        data_path = optarg;
        break;
      case 'h':
        hostname = optarg;
        break;
      case 'p':
        port = std::stoi(optarg);
        break;
      case 'm':
        mode = optarg;
        if (mode != "publish" && mode != "subscribe") {
          HOLOSCAN_LOG_ERROR("Invalid mode '{}'. Must be 'publish' or 'subscribe'", mode);
          return false;
        }
        break;
      case '?':
      default:
        std::cout << "UCXX Endoscopy Tool Tracking - Distributed Application\n"
                  << "\n"
                  << "Usage: " << argv[0] << " [options]\n"
                  << "\n"
                  << "Options:\n"
                  << "  -d, --data <path>        Path to data directory (required for publisher)\n"
                  << "  -c, --config <path>      Path to config file (optional)\n"
                  << "  -h, --hostname <host>    Hostname (default: 0.0.0.0 for publisher,\n"
                  << "                           127.0.0.1 for subscriber)\n"
                  << "  -p, --port <port>        Port number (default: 50008)\n"
                  << "  -m, --mode <mode>        Mode: 'publish' or 'subscribe' (required)\n"
                  << "  -?, --help               Show this help message\n"
                  << "\n"
                  << "Description:\n"
                  << "  Publisher: Processes video, renders with overlays, and broadcasts\n"
                  << "          rendered frames\n"
                  << "  Subscriber: Receives pre-rendered frames from publisher and displays them\n"
                  << "\n"
                  << "Examples:\n"
                  << "  Publisher: " << argv[0] << " --mode publish --data /path/to/data\n"
                  << "  Subscriber: " << argv[0] << " --mode subscribe --hostname publisher_ip\n"
                  << "\n";
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
  std::string hostname = "";
  int port = 50008;
  std::string mode = "";

  if (!parse_arguments(argc, argv, data_directory, config_path, hostname, port, mode)) {
    return 1;
  }

  // Validate mode is specified
  if (mode.empty()) {
    HOLOSCAN_LOG_ERROR("Mode must be specified. Use --mode publish or --mode subscribe");
    return 1;
  }

  // Set default hostname based on mode if not specified
  if (hostname.empty()) {
    if (mode == "subscribe") {
      hostname = "127.0.0.1";  // Default to localhost for subscriber
    } else {
      hostname = "0.0.0.0";  // Default to all interfaces for publisher
    }
  }

  // For publisher mode, validate data directory
  if (mode == "publish") {
    if (data_directory.empty()) {
      // Get the input data environment variable
      auto input_path = std::getenv("HOLOSCAN_INPUT_PATH");
      if (input_path != nullptr && input_path[0] != '\0') {
        data_directory = std::string(input_path) + "/endoscopy";
      } else if (std::filesystem::is_directory(
                     std::filesystem::current_path() / "data/endoscopy")) {
        data_directory = std::string(
            (std::filesystem::current_path() / "data/endoscopy").c_str());
      } else {
        HOLOSCAN_LOG_ERROR(
            "Data directory required for publisher mode. Use --data or set HOLOSCAN_INPUT_PATH");
        return 1;
      }
    }
  }

  // Get application path for config
  std::string app_path(PATH_MAX, '\0');
  if (readlink("/proc/self/exe", app_path.data(), app_path.size() - 1) == -1) {
    HOLOSCAN_LOG_ERROR("Failed to get the application path");
    return 1;
  }
  app_path = std::filesystem::canonical(app_path).parent_path();

  // Determine config path
  if (config_path.empty()) {
    auto config_file_path = std::getenv("HOLOSCAN_CONFIG_PATH");
    if (config_file_path == nullptr || config_file_path[0] == '\0') {
      config_path = app_path / std::filesystem::path("ucxx_endoscopy_tool_tracking.yaml");

      // Fallback to standard config if UCXX-specific config doesn't exist
      if (!std::filesystem::exists(config_path)) {
        config_path = app_path / std::filesystem::path("endoscopy_tool_tracking.yaml");
      }
    } else {
      config_path = config_file_path;
    }
  }

  HOLOSCAN_LOG_INFO("=== UCXX Endoscopy Tool Tracking ===");
  HOLOSCAN_LOG_INFO("Mode: {}", mode);
  HOLOSCAN_LOG_INFO("Hostname: {}", hostname);
  HOLOSCAN_LOG_INFO("Port: {}", port);
  if (mode == "publish") {
    HOLOSCAN_LOG_INFO("Data path: {}", data_directory);
  }
  HOLOSCAN_LOG_INFO("Config: {}", config_path);

  // Launch appropriate application based on mode
  if (mode == "publish") {
    auto app = holoscan::make_application<holoscan::apps::UcxxEndoscopyPublisherApp>();
    app->config(config_path);
    app->set_datapath(data_directory);
    app->set_hostname(hostname);
    app->set_port(port);

    HOLOSCAN_LOG_INFO("Starting PUBLISHER: Processing video and broadcasting to subscribers");
    app->run();

  } else if (mode == "subscribe") {
    auto app = holoscan::make_application<holoscan::apps::UcxxEndoscopySubscriberApp>();
    app->config(config_path);
    app->set_hostname(hostname);
    app->set_port(port);

    HOLOSCAN_LOG_INFO("Starting SUBSCRIBER: Receiving processed frames from publisher");
    app->run();
  }

  HOLOSCAN_LOG_INFO("Application completed successfully");
  return 0;
}
