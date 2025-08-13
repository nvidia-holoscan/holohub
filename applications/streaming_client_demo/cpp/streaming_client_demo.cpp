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

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include "streaming_client.hpp"



// Create a default YAML configuration file if it doesn't exist
bool ensure_config_file_exists(const std::string& config_path) {
  std::ifstream file(config_path);
  if (file.good()) {
    std::cout << "Found existing config file: " << config_path << std::endl;
    file.close();
    return true;
  }

  std::cout << "Config file not found, creating default at: " << config_path << std::endl;

  std::ofstream out_file(config_path);
  if (!out_file.is_open()) {
    std::cerr << "Error: Could not create config file at " << config_path << std::endl;
    return false;
  }

  // Write default YAML configuration
  out_file << "%YAML 1.2\n";
  out_file << "---\n";
  out_file << "# Application configuration\n";
  out_file << "application:\n";
  out_file << "  title: Streaming Client Test App\n";
  out_file << "  version: 1.0\n";
  out_file << "  log_level: INFO\n\n";

  out_file << "# Streaming client settings\n";
  out_file << "streaming_client:\n";
  out_file << "  # Video/stream parameters\n";
  out_file << "  width: 854\n";
  out_file << "  height: 480\n";
  out_file << "  fps: 30\n";
  out_file << "  \n";
  out_file << "  # Server connection settings\n";
  out_file << "  server_ip: \"127.0.0.1\"\n";
  out_file << "  signaling_port: 48010\n";
  out_file << "  \n";
  out_file << "  # Operation mode\n";
  out_file << "  receive_frames: true\n";
  out_file << "  send_frames: true\n";
  out_file << "  \n";
  out_file << "  # Advanced options\n";
  out_file << "  timeout_ms: 5000\n";
  out_file << "  reconnect_attempts: 3\n";
  out_file << "  buffer_size: 10\n\n";

  out_file << "# Visualization options\n";
  out_file << "visualize_frames: true\n\n";

  out_file << "# HoloViz configuration (used only if visualize_frames is true)\n";
  out_file << "holoviz:\n";
  out_file << "  # Window size and title\n";
  out_file << "  width: 854\n";
  out_file << "  height: 480\n";
  out_file << "  window_title: \"Streaming Client Test\"\n";
  out_file << "  \n";
  out_file << "  # Rendering parameters\n";
  out_file << "  color_format: \"rgb\"\n";
  out_file << "  enable_render_buffer_timestamp: true\n\n";

  out_file << "# Video replayer configuration\n";
  out_file << "replayer:\n";
  out_file << "  basename: \"surgical_video\"\n";
  out_file << "  frame_rate: 30\n";
  out_file << "  repeat: true\n";
  out_file << "  realtime: true\n";
  out_file << "  count: 0\n\n";

  out_file << "# Scheduler configuration (optional)\n";
  out_file << "scheduler: \"default\"\n";

  out_file.close();
  std::cout << "Created default config file at: " << config_path << std::endl;
  return true;
}

class StreamingClientTestApp : public holoscan::Application {
 public:
  void set_width(uint32_t width) { width_ = width; }
  void set_height(uint32_t height) { height_ = height; }
  void set_fps(uint32_t fps) { fps_ = fps; }
  void set_server_ip(const std::string& server_ip) { server_ip_ = server_ip; }
  void set_signaling_port(uint16_t signaling_port) { signaling_port_ = signaling_port; }
  void set_receive_frames(bool receive_frames) { receive_frames_ = receive_frames; }
  void set_send_frames(bool send_frames) { send_frames_ = send_frames; }
  void set_visualize_frames(bool visualize_frames) { visualize_frames_ = visualize_frames; }
  void set_datapath(const std::string& datapath) { datapath_ = datapath; }

  void compose() override {
    using namespace holoscan;

    auto allocator = make_resource<UnboundedAllocator>("allocator");

    // Find a valid data directory
    std::string data_path = datapath_;
    if (!std::filesystem::exists(data_path)) {
      HOLOSCAN_LOG_WARN("Data directory '{}' does not exist!", data_path);

      // Try alternative paths
      if (std::filesystem::exists("/workspace/holoscan-sdk/data")) {
        data_path = "/workspace/holoscan-sdk/data";
        HOLOSCAN_LOG_INFO("Using SDK data path: {}", data_path);
      } else if (std::filesystem::exists("/workspace/holohub/data")) {
        data_path = "/workspace/holohub/data";
        HOLOSCAN_LOG_INFO("Using HoloHub data path: {}", data_path);
      } else if (std::filesystem::exists("data")) {
        data_path = "data";
        HOLOSCAN_LOG_INFO("Using local data path: {}", data_path);
      } else {
        HOLOSCAN_LOG_ERROR(
            "No valid data directory found! Please set HOLOSCAN_INPUT_PATH or provide a "
            "valid data directory.");
        return;
      }
    }

    // Log the full path we're trying to use
    std::string full_video_path = data_path + "/surgical_video";
    HOLOSCAN_LOG_INFO("Attempting to load video from: {}", full_video_path);

    // Fixed memory pool sizing for uint8 BGR data
    uint64_t source_block_size = width_ * height_ * 3;  // 3 channels for BGR
    uint64_t source_num_blocks = 4;  // Keep multiple blocks for pipeline stability

    auto source = make_operator<ops::VideoStreamReplayerOp>(
        "replayer",
        from_config("replayer"),
        Arg("directory", data_path),  // Use the resolved data_path
        Arg("count", static_cast<int64_t>(0)));

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter",
        from_config("format_converter"),
        Arg("pool", make_resource<BlockMemoryPool>(
            "pool", 1, source_block_size, source_num_blocks)),
        Arg("cuda_stream_pool", cuda_stream_pool));

    auto streaming_client = make_operator<ops::StreamingClientOp>(
        "streaming_client",
        Arg("width", width_),
        Arg("height", height_),
        Arg("fps", fps_),
        Arg("server_ip", server_ip_),
        Arg("signaling_port", signaling_port_),
        Arg("receive_frames", receive_frames_),
        Arg("send_frames", send_frames_));

    add_flow(source, format_converter, {{"output", "source_video"}});
    add_flow(format_converter, streaming_client);

    if (visualize_frames_) {
        auto holoviz = make_operator<ops::HolovizOp>(
            "holoviz",
            from_config("holoviz"),
            Arg("width", width_),
            Arg("height", height_),
            Arg("allocator", allocator),
            Arg("cuda_stream_pool", cuda_stream_pool));

        add_flow(streaming_client, holoviz, {{"output", "render"}});
    }
  }

 private:
  // Default parameters (will be overridden from config if available)
  uint32_t width_ = 854;
  uint32_t height_ = 480;
  uint32_t fps_ = 30;
  std::string server_ip_ = "127.0.0.1";
  uint16_t signaling_port_ = 48010;
  bool receive_frames_ = true;
  bool send_frames_ = true;
  bool visualize_frames_ = true;
  std::string source_ = "replayer";  // Added source_ member variable
  std::string datapath_ = "data/endoscopy";
  bool gpu_tensor_ = false;
  int64_t count_ = 10;
  int32_t batch_size_ = 0;
  int32_t rows_ = 32;
  int32_t columns_ = 64;
  int32_t channels_ = 0;
  std::string data_type_{"uint8_t"};

};

void print_usage() {
  std::cout << "Usage: streaming_client_demo [options]\n"
            << "  -h, --help                Show this help message\n"
            << "  -c, --config <file>        Configuration file path "
            << "(default: streaming_client_demo.yaml)\n"
            << "  -d, --data <directory>     Data directory (default: environment "
            << "variable HOLOSCAN_INPUT_PATH or current directory)\n"
            << std::endl;
}

// Helper function to safely get config values with defaults
template<typename T>
T get_config_value(holoscan::Application* app, const std::string& key,
                   const T& default_value) {
  try {
    return app->from_config(key).as<T>();
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to read config value '" << key << "': " << e.what() << std::endl;
    std::cerr << "Using default value: " << default_value << std::endl;
    return default_value;
  }
}

// Specialization for std::string to avoid printing issues
template<>
std::string get_config_value<std::string>(holoscan::Application* app,
                                           const std::string& key,
                                           const std::string& default_value) {
  try {
    return app->from_config(key).as<std::string>();
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to read config value '" << key << "': " << e.what()
              << std::endl;
    std::cerr << "Using default value: " << default_value << std::endl;
    return default_value;
  }
}

int main(int argc, char** argv) {
  // Default config file path
  std::string config_path = "streaming_client_demo.yaml";
  std::string data_directory = "";

  // Define command line options
  static struct option long_options[] = {
      {"config", required_argument, 0, 'c'},
      {"data", required_argument, 0, 'd'},
      {"help", no_argument, 0, '?'},
      {0, 0, 0, 0}
  };

  // Parse command line arguments
  int opt;
  int option_index = 0;
  while ((opt = getopt_long(argc, argv, "c:d:?", long_options, &option_index)) != -1) {
    switch (opt) {
      case 'c':
        config_path = optarg;
        break;
      case 'd':
        data_directory = optarg;
        break;
      case '?':
        print_usage();
        return 0;
      default:
        print_usage();
        return 1;
    }
  }

  // Check for data directory
  if (data_directory.empty()) {
    // Try multiple possible locations
    const std::vector<std::string> possible_paths = {
      std::string(std::getenv("HOLOSCAN_INPUT_PATH") ? std::getenv("HOLOSCAN_INPUT_PATH") : ""),
      std::string((std::filesystem::current_path() / "data").c_str()),
      std::string((std::filesystem::current_path() / "data/endoscopy").c_str()),
      "/workspace/holoscan-sdk/data",
      "/workspace/holoscan-sdk/data/endoscopy",
      "/workspace/holohub/data/endoscopy",
      "/workspace/holohub/data",
      "/opt/nvidia/holoscan/data",
      "/opt/nvidia/holoscan/data/endoscopy",
      "/workspace/holohub-internal/data",
      "/workspace/holohub-internal/data/endoscopy"
    };

    for (const auto& path : possible_paths) {
      if (!path.empty() && std::filesystem::exists(path)) {
        // Check if the video file exists in this directory
        std::string video_path = path + "/surgical_video.gxf_index";
        if (std::filesystem::exists(video_path)) {
          data_directory = path;
          std::cout << "Found valid data directory with video file: " << data_directory
                    << std::endl;
          break;
        } else {
          std::cout << "Directory exists but no video file found at: " << video_path
                    << std::endl;
        }
      }
    }

    if (data_directory.empty()) {
      std::cerr << "ERROR: Could not find surgical_video.gxf_index in any of the "
                << "standard locations." << std::endl;
      std::cerr << "Please ensure the video file is present in one of these locations:"
                << std::endl;
      for (const auto& path : possible_paths) {
        if (!path.empty()) {
          std::cerr << "  - " << path << std::endl;
        }
      }
      return 1;  // Exit with error
    }
  }

  // Verify both the directory and video file exist
  if (!std::filesystem::exists(data_directory)) {
    std::cerr << "ERROR: Specified data directory '" << data_directory
              << "' does not exist!" << std::endl;
    return 1;
  }

  std::string video_path = data_directory + "/surgical_video.gxf_index";
  if (!std::filesystem::exists(video_path)) {
    std::cerr << "ERROR: Video file not found at: " << video_path << std::endl;
    return 1;
  }

  // Print current working directory and found paths for debugging
  char cwd[1024];
  if (getcwd(cwd, sizeof(cwd)) != NULL) {
    std::cout << "Current working directory: " << cwd << std::endl;
  }
  std::cout << "Using data directory: " << data_directory << std::endl;
  std::cout << "Video file path: " << video_path << std::endl;

  // Create a default configuration file if it doesn't exist
  if (!ensure_config_file_exists(config_path)) {
    std::cerr << "WARNING: Failed to create default configuration file." << std::endl;
    std::cerr << "Will try to run with built-in defaults." << std::endl;
  }

  // Create the application
  auto app = holoscan::make_application<StreamingClientTestApp>();

  // Print configuration file path
  std::cout << "Streaming Client Test Application\n"
            << "Using config file: " << config_path << std::endl;

  // Try to load configuration from YAML (but continue even if it fails)
  try {
    app->config(config_path);
    std::cout << "Successfully loaded configuration from: " << config_path << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to load config file: " << e.what() << std::endl;
    std::cerr << "Will continue with default values" << std::endl;
  }

  // Load parameters from config with safe defaults
  uint32_t width = get_config_value(app.get(), "streaming_client.width", 854U);
  uint32_t height = get_config_value(app.get(), "streaming_client.height", 480U);
  uint32_t fps = get_config_value(app.get(), "streaming_client.fps", 30U);
  std::string server_ip = get_config_value(app.get(), "streaming_client.server_ip",
                                           std::string("127.0.0.1"));
  uint16_t signaling_port = get_config_value(app.get(),
                                             "streaming_client.signaling_port", 48010);
  bool receive_frames = get_config_value(app.get(), "streaming_client.receive_frames",
                                         true);
  bool send_frames = get_config_value(app.get(), "streaming_client.send_frames", true);
  bool visualize_frames = get_config_value(app.get(), "visualize_frames", true);

  // Set application parameters from config
  app->set_width(width);
  app->set_height(height);
  app->set_fps(fps);
  app->set_server_ip(server_ip);
  app->set_signaling_port(signaling_port);
  app->set_receive_frames(receive_frames);
  app->set_send_frames(send_frames);
  app->set_visualize_frames(visualize_frames);

  // Set data directory
  std::cout << "Using data from: " << data_directory << std::endl;
  app->set_datapath(data_directory);

  std::cout << "Configuration:\n"
            << "- Resolution: " << width << "x" << height << "\n"
            << "- FPS: " << fps << "\n"
            << "- Server: " << server_ip << ":" << signaling_port << "\n"
            << "- Receive frames: " << (receive_frames ? "yes" : "no") << "\n"
            << "- Send frames: " << (send_frames ? "yes" : "no") << "\n"
            << "- Visualize frames: " << (visualize_frames ? "yes" : "no") << std::endl;

  // Configure scheduler based on the config file
  std::string scheduler = get_config_value(app.get(), "scheduler", std::string("default"));
  if (scheduler == "multi_thread") {
    // Use MultiThreadScheduler with minimal configuration
    app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>("multithread-scheduler"));
    std::cout << "Using multi-thread scheduler" << std::endl;
  } else if (scheduler == "event_based") {
    // Use EventBasedScheduler with minimal configuration
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>("event-based-scheduler"));
    std::cout << "Using event-based scheduler" << std::endl;
  } else if (scheduler == "greedy") {
    // Use GreedyScheduler with minimal configuration
    app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>("greedy-scheduler"));
    std::cout << "Using greedy scheduler" << std::endl;
  } else {
    // Always fallback to default scheduler
    std::cout << "Using default scheduler" << std::endl;
  }

  // Turn on data flow tracking if specified in the YAML
  bool tracking = get_config_value(app.get(), "tracking", false);
  holoscan::DataFlowTracker* tracker = nullptr;
  if (tracking) {
    std::cout << "Enabling data flow tracking" << std::endl;
    tracker = &app->track(0, 0, 0);
  }

  // Run the application
  app->run();

  // Print data flow tracking results if enabled
  if (tracking && tracker) {
    tracker->print();
  }

  return 0;
}
