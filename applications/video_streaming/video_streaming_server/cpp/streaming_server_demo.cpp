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

#include "streaming_server_op.hpp"

namespace fs = std::filesystem;


namespace holoscan::ops {



}  // namespace holoscan::ops

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
  out_file << "  title: Streaming Server Test App\n";
  out_file << "  version: 1.0\n";
  out_file << "  log_level: INFO\n\n";

  out_file << "# Streaming server settings\n";
  out_file << "streaming_server:\n";
  out_file << "  # Video/stream parameters\n";
  out_file << "  width: 854\n";
  out_file << "  height: 480\n";
  out_file << "  fps: 30\n";
  out_file << "  \n";
  out_file << "  # Server connection settings\n";
  out_file << "  server_ip: \"127.0.0.1\"\n";
  out_file << "  port: 48010\n";
  out_file << "  \n";
  out_file << "  # Operation mode - Bidirectional streaming\n";
  out_file << "  receive_frames: true\n";
  out_file << "  send_frames: true\n";
  out_file << "  visualize_frames: false\n";
  out_file << "  \n";
  out_file << "  # Advanced options\n";
  out_file << "  timeout_ms: 5000\n";
  out_file << "  reconnect_attempts: 3\n";
  out_file << "  buffer_size: 10\n\n";

  out_file << "# Visualization options (disabled by default for server mode)\n";
  out_file << "visualize_frames: false\n\n";

  out_file << "# HoloViz configuration (used only if visualize_frames is true)\n";
  out_file << "holoviz:\n";
  out_file << "  # Window size and title\n";
  out_file << "  width: 854\n";
  out_file << "  height: 480\n";
  out_file << "  window_title: \"Streaming Server Test\"\n";
  out_file << "  \n";
  out_file << "  # Rendering parameters\n";
  out_file << "  color_format: \"rgb\"\n";
  out_file << "  enable_render_buffer_timestamp: true\n\n";

  out_file << "# Video replayer configuration\n";
  out_file << "replayer:\n";
  out_file << "  basename: \"surgical_video\"\n";
  out_file << "  frame_rate: 30\n";
  out_file << "  repeat: true\n\n";

  out_file << "# Scheduler configuration (optional)\n";
  out_file << "scheduler: \"greedy\"\n";

  out_file << "# Data flow tracking (optional)\n";
  out_file << "tracking: false\n";

  out_file.close();
  std::cout << "Created default config file at: " << config_path << std::endl;
  return true;
}

class StreamingServerTestApp : public holoscan::Application {
 public:
  void set_width(uint32_t width) { width_ = width; }
  void set_height(uint32_t height) { height_ = height; }
  void set_fps(uint32_t fps) { fps_ = fps; }
  void set_server_ip(const std::string& server_ip) { server_ip_ = server_ip; }
  void set_port(uint16_t port) { port_ = port; }
  void set_receive_frames(bool receive_frames) { receive_frames_ = receive_frames; }
  void set_send_frames(bool send_frames) { send_frames_ = send_frames; }
  void set_visualize_frames(bool visualize_frames) { visualize_frames_ = visualize_frames; }
  void set_datapath(const std::string& datapath) { datapath_ = datapath; }

  void compose() override {
    using namespace holoscan;

    // Check if we have video data available for functional testing
    fs::path data_dir(datapath_);
    fs::path fallback_dir("/workspace/holohub/data");
    bool has_video_data = false;
    fs::path final_data_dir;

    // Check if original data directory exists and has video files
    if (fs::exists(data_dir) && fs::exists(data_dir / "surgical_video.gxf_index")) {
      HOLOSCAN_LOG_INFO("🎬 FUNCTIONAL test: Using real video data from {}", data_dir.string());
      final_data_dir = data_dir;
      has_video_data = true;
    } else if (fs::exists(fallback_dir) && fs::exists(fallback_dir / "surgical_video.gxf_index")) {
      HOLOSCAN_LOG_INFO("🔧 INFRASTRUCTURE test: No video data found, testing StreamingServer functionality only");
      HOLOSCAN_LOG_INFO("Found valid data directory with video file: {}", fallback_dir.string());
      HOLOSCAN_LOG_INFO("Using data directory: {}", fallback_dir.string());
      HOLOSCAN_LOG_INFO("Video file path: {}/surgical_video.gxf_index", fallback_dir.string());
      final_data_dir = fallback_dir;
      has_video_data = true;
    }

    if (has_video_data && !datapath_.empty()) {
      // Functional test mode: Create video pipeline
      HOLOSCAN_LOG_INFO("🎬 Using real video data from: {}", final_data_dir.string());

      // Create allocator
      auto allocator = make_resource<UnboundedAllocator>("allocator");

      // Video source with real data
      auto source = make_operator<ops::VideoStreamReplayerOp>(
          "video_source",
          Arg("directory", final_data_dir.string()),
          Arg("basename", "surgical_video"),
          Arg("frame_rate", 30.0f),
          Arg("realtime", false),
          Arg("repeat", false),
          Arg("count", static_cast<int64_t>(50))  // Process 50 frames for testing
      );

      // Format converter
      auto format_converter = make_operator<ops::FormatConverterOp>(
          "format_converter",
          Arg("pool", allocator),
          Arg("out_dtype", "uint8"),
          Arg("out_channel_order", std::vector<int>{2, 1, 0})  // BGR
      );

      // Streaming server operator
      auto streaming_server = make_operator<ops::StreamingServerOp>(
          "streaming_server", from_config("streaming_server"));

      // Connect the video pipeline to streaming server
      add_flow(source, format_converter, {{"output", "source_video"}});
      add_flow(format_converter, streaming_server, {{"tensor", "input_frames"}});

      HOLOSCAN_LOG_INFO("Application composed with video pipeline: VideoSource → FormatConverter → StreamingServer");
    } else {
      // Infrastructure test mode: Standalone streaming server
      auto streaming_server = make_operator<ops::StreamingServerOp>(
          "streaming_server", from_config("streaming_server"));

      // Add the operator to the pipeline
      add_operator(streaming_server);

      HOLOSCAN_LOG_INFO("Application composed with streaming server using standalone mode (no video pipeline)");
    }
  }

 private:
  // Default parameters (will be overridden from config if available)
  uint32_t width_ = 854;
  uint32_t height_ = 480;
  uint32_t fps_ = 30;
  std::string server_ip_ = "127.0.0.1";
  uint16_t port_ = 48010;  // Changed to match YAML config and client
  bool receive_frames_ = true;
  bool send_frames_ = true;
  bool visualize_frames_ = false;
  std::string source_ = "replayer";  // Added source_ member variable
  std::string datapath_ = "data/endoscopy";
  bool gpu_tensor_ = false;
  int64_t count_ = 10;
  int32_t batch_size_ = 0;
  int32_t rows_ = 32;
  int32_t columns_ = 64;
  int32_t channels_ = 0;
  std::string data_type_{"uint8_t"};
  // Added datapath_ member variable
};

void print_usage() {
  std::cout << "Usage: streaming_server_demo [options]\n"
            << "Options:\n"
            << "  -c, --config <file>        Configuration file path (default: "
               "streaming_server_demo.yaml)\n"
            << "  -d, --data <directory>     Data directory (default: environment variable "
               "HOLOSCAN_INPUT_PATH or current directory)\n"
            << "  -?, --help                 Show this help message\n"
            << std::endl;
}

// Helper function to safely get config values with defaults
template <typename T>
T get_config_value(holoscan::Application* app, const std::string& key, const T& default_value) {
  try {
    return app->from_config(key).as<T>();
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to read config value '" << key << "': " << e.what() << std::endl;
    std::cerr << "Using default value: " << default_value << std::endl;
    return default_value;
  }
}

// Specialization for std::string to avoid printing issues
template <>
std::string get_config_value<std::string>(holoscan::Application* app, const std::string& key,
                                          const std::string& default_value) {
  try {
    return app->from_config(key).as<std::string>();
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to read config value '" << key << "': " << e.what() << std::endl;
    std::cerr << "Using default value: " << default_value << std::endl;
    return default_value;
  }
}

// Specialization for uint16_t to ensure proper port handling
template <>
uint16_t get_config_value<uint16_t>(holoscan::Application* app, const std::string& key,
                                    const uint16_t& default_value) {
  try {
    // Try to read as int first, then convert to uint16_t
    int port_value = app->from_config(key).as<int>();
    if (port_value < 0 || port_value > 65535) {
      std::cerr << "Warning: Port value " << port_value
                << " out of range for uint16_t. Using default: " << default_value << std::endl;
      return default_value;
    }
    return static_cast<uint16_t>(port_value);
  } catch (const std::exception& e) {
    std::cerr << "Warning: Failed to read config value '" << key << "': " << e.what() << std::endl;
    std::cerr << "Using default value: " << default_value << std::endl;
    return default_value;
  }
}

int main(int argc, char** argv) {
  // Default config file path
  std::string config_path = "streaming_server_demo.yaml";
  std::string data_directory = "";

  // Parse command line arguments
  static struct option long_options[] = {{"config", required_argument, 0, 'c'},
                                         {"data", required_argument, 0, 'd'},
                                         {"help", no_argument, 0, '?'},
                                         {0, 0, 0, 0}};

  int option_index = 0;
  int c;
  while ((c = getopt_long(argc, argv, "c:d:?", long_options, &option_index)) != -1) {
    switch (c) {
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
        std::cerr << "Invalid option. Use --help for usage information." << std::endl;
        return 1;
    }
  }

  // Create a default configuration file if it doesn't exist
  if (!ensure_config_file_exists(config_path)) {
    std::cerr << "WARNING: Failed to create default configuration file." << std::endl;
    std::cerr << "Will try to run with built-in defaults." << std::endl;
  }

  // Create the application with error handling
  std::shared_ptr<StreamingServerTestApp> app;
  try {
    app = holoscan::make_application<StreamingServerTestApp>();

    // Print configuration file path
    std::cout << "Streaming Server Test Application\n"
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
    uint32_t width = get_config_value(app.get(), "streaming_server.width", 854u);
    uint32_t height = get_config_value(app.get(), "streaming_server.height", 480u);
    uint32_t fps = get_config_value(app.get(), "streaming_server.fps", 30u);
    std::string server_ip = get_config_value(app.get(), "streaming_server.server_ip",
                                           std::string("127.0.0.1"));
    uint16_t port = get_config_value(app.get(), "streaming_server.port", 48010);
    bool receive_frames = get_config_value(app.get(), "streaming_server.receive_frames", true);
    bool send_frames = get_config_value(app.get(), "streaming_server.send_frames", true);
    bool visualize_frames = get_config_value(app.get(), "streaming_server.visualize_frames", false);

    // Set application parameters from config
    app->set_width(width);
    app->set_height(height);
    app->set_fps(fps);
    app->set_server_ip(server_ip);
    app->set_port(port);
    app->set_receive_frames(receive_frames);
    app->set_send_frames(send_frames);
    app->set_visualize_frames(visualize_frames);
    app->set_datapath(data_directory);

    std::cout << "Using data from: " << (data_directory.empty() ? "none (standalone mode)" : data_directory) << std::endl;
    std::cout << "Configuration:\n"
              << "- Resolution: " << width << "x" << height << "\n"
              << "- FPS: " << fps << "\n"
              << "- Server: " << server_ip << ":" << port << "\n"
              << "- Receive frames: " << (receive_frames ? "yes" : "no") << "\n"
              << "- Send frames: " << (send_frames ? "yes" : "no") << "\n"
              << "- Visualize frames: " << (visualize_frames ? "yes" : "no") << std::endl;

    // Configure scheduler
    std::string scheduler = get_config_value(app.get(), "scheduler", std::string("greedy"));
    if (scheduler == "multi_thread") {
      app->scheduler(app->make_scheduler<holoscan::MultiThreadScheduler>("multithread-scheduler"));
      std::cout << "Using multi-thread scheduler" << std::endl;
    } else if (scheduler == "event_based") {
      app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>("event-based-scheduler"));
      std::cout << "Using event-based scheduler" << std::endl;
    } else {
      app->scheduler(app->make_scheduler<holoscan::GreedyScheduler>("greedy-scheduler"));
      std::cout << "Using greedy scheduler" << std::endl;
    }

    // Enable data flow tracking if specified
    bool tracking = get_config_value(app.get(), "tracking", false);
    holoscan::DataFlowTracker* tracker = nullptr;
    if (tracking) {
      std::cout << "Enabling data flow tracking" << std::endl;
      tracker = &app->track(0, 0, 0);
    }

    std::cout << "Starting streaming server..." << std::endl;
    std::cout << "Press Ctrl+C to stop gracefully" << std::endl;

    // Run the application with exception handling
    app->run();

    // Print data flow tracking results if enabled
    if (tracking && tracker) {
      tracker->print();
    }

    std::cout << "Server stopped successfully" << std::endl;
  } catch (const std::exception& e) {
    std::cerr << "Application error: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown application error" << std::endl;
    return 1;
  }

  return 0;
}
