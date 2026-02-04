/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

/**
 * @file main.cpp
 * @brief HoloCat Application Entry Point
 *
 * Main entry point for the HoloCat EtherCAT real-time integration application.
 * Handles command line argument parsing, configuration loading, and application
 * lifecycle management.
 */

// System includes
#include <cstring>
#include <filesystem>
#include <iostream>

// Third-party includes
#include <holoscan/holoscan.hpp>

// Local includes
#include "holocat_app.hpp"

/**
 * @brief Command line arguments structure
 */
struct CommandLineArgs {
  std::string config_file;
  bool print_config_only = false;
  bool show_help = false;
};

/**
 * @brief Print application usage information
 * @param program_name Name of the program executable
 */
void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
  std::cout << "HoloCat - EtherCAT Real-time Integration with Holoscan SDK\n\n";
  std::cout << "Options:\n";
  std::cout << "  -c, --config FILE    Load configuration from YAML file\n";
  std::cout << "  -h, --help          Show this help message\n";
  std::cout << "  --print-config      Print loaded configuration and exit\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << program_name << " --config /path/to/config.yaml\n";
  std::cout << "  " << program_name << " --print-config\n\n";
}


/**
 * @brief Parse command line arguments
 * @param argc Argument count
 * @param argv Argument values
 * @return Parsed command line arguments structure
 * @throws std::runtime_error if parsing fails
 */
CommandLineArgs parse_arguments(int argc, char** argv) {
  CommandLineArgs args;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      args.show_help = true;
    } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--config") == 0) {
      if (i + 1 < argc) {
        args.config_file = argv[++i];
      } else {
        throw std::runtime_error("--config requires a file path");
      }
    } else if (strcmp(argv[i], "--print-config") == 0) {
      args.print_config_only = true;
    } else {
      throw std::runtime_error(std::string("Unknown option: ") + argv[i]);
    }
  }

  return args;
}

int main(int argc, char** argv) {
  HOLOSCAN_LOG_INFO("HoloCat - EtherCAT Real-time Integration");
  HOLOSCAN_LOG_INFO("Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES");

  try {
    // Parse command line arguments
    CommandLineArgs args = parse_arguments(argc, argv);

    // Handle help request
    if (args.show_help) {
      print_usage(argv[0]);
      return 0;
    }

    // Create and configure the Holoscan application
    auto app = holoscan::make_application<holocat::HolocatApp>();

    // Load configuration file
    std::filesystem::path config_path;
    if (!args.config_file.empty()) {
      config_path = args.config_file;
    } else {
      config_path =
          std::filesystem::canonical(argv[0]).parent_path() / "holocat_config.yaml";
    }

    if (!std::filesystem::exists(config_path)) {
      throw std::runtime_error("Configuration file not found: " + config_path.string());
    }

    app->config(config_path.string());

    // Handle configuration printing request
    if (args.print_config_only) {
      try {
        // Note: extract_config() is intentionally called here (rather than in compose()). Move it?
        auto config = app->extract_config();
        std::cout << "HoloCat Configuration:\n";
        std::cout << "  Adapter: " << config.adapter_name << "\n";
        std::cout << "  ENI File: " << config.eni_file << "\n";
        std::cout << "  Cycle Time: " << config.cycle_time_us << " μs\n";
        std::cout << "  RT Priority: " << config.rt_priority << "\n";
        // ... print other fields
      } catch (const std::exception& e) {
        throw std::runtime_error("Configuration error: " + std::string(e.what()));
      }
      return 0;
    }

    // Run the main application
    HOLOSCAN_LOG_INFO("Starting HoloCat application...");
    app->run();
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application error: {}", e.what());
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  HOLOSCAN_LOG_INFO("HoloCat application finished");
  return 0;
}

