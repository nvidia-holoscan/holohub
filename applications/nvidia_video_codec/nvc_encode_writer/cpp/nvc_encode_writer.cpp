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

#include <filesystem>
#include <getopt.h>
#include <iostream>
#include <string>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

// Include the NVIDIA Video Codec operators
#include "nv_video_encoder.hpp"
#include "tensor_to_file.hpp"

class NVIDIAVideoCodecApp : public holoscan::Application {
 public:
  void set_data_path(const std::string& path) { data_path_ = path; }

  void compose() override {
    using namespace holoscan;

    // Get configuration parameters
    uint32_t width = from_config("video.width").as<uint32_t>();
    uint32_t height = from_config("video.height").as<uint32_t>();
    uint64_t source_block_size = width * height * 3 * 4;
    uint64_t source_num_blocks = 2;

    // Validate data path
    if (!std::filesystem::exists(data_path_)) {
      throw std::runtime_error("Could not find video data: " + data_path_);
    }

    // Create Video Stream Replayer
    auto source = make_operator<ops::VideoStreamReplayerOp>(
        "replayer",
        Arg("directory", data_path_),
        Arg("allocator", make_resource<RMMAllocator>("video_replayer_allocator")),
        from_config("replayer"));

    // Create Format Converter
    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter",
        Arg("pool") = make_resource<BlockMemoryPool>(
                        "pool",
                        static_cast<int32_t>(MemoryStorageType::kDevice),
                        source_block_size,
                        source_num_blocks),
        from_config("format_converter"));

    // Create NVIDIA Video Encoder
    auto encoder = make_operator<ops::NvVideoEncoderOp>(
        "nv_encoder",
        Arg("width", width),
        Arg("height", height),
        Arg("allocator") = make_resource<BlockMemoryPool>(
                             "encoder_pool",
                             static_cast<int32_t>(MemoryStorageType::kHost),
                             source_block_size,
                             source_num_blocks),
        from_config("encoder"));

    // Create Tensor to File Writer
    auto writer = make_operator<ops::TensorToFileOp>(
        "nv_writer",
        Arg("allocator", make_resource<RMMAllocator>("video_writer_allocator")),
        from_config("writer"));

    // Add flows
    add_flow(source, format_converter, {{"output", "source_video"}});
    add_flow(format_converter, encoder, {{"tensor", "input"}});
    add_flow(encoder, writer, {{"output", "input"}});
  }

 private:
  std::string data_path_ = "";
};

void print_usage(const char* program_name) {
  std::cout << "Usage: " << program_name << " [options]\n";
  std::cout << "Options:\n";
  std::cout << "  -h, --help      Show this help message\n";
  std::cout << "  -c, --config    Set config path to override the default config file location\n";
  std::cout << "  -d, --data      Set the data path\n";
}

bool parse_arguments(int argc, char** argv, std::string& config_path, std::string& data_path) {
  static struct option long_options[] = {
      {"help", no_argument, 0, 'h'},
      {"config", required_argument, 0, 'c'},
      {"data", required_argument, 0, 'd'},
      {0, 0, 0, 0}};

  int option_index = 0;
  int c;

  while ((c = getopt_long(argc, argv, "hc:d:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'h':
        print_usage(argv[0]);
        return false;
      case 'c':
        config_path = optarg;
        break;
      case 'd':
        data_path = optarg;
        break;
      case '?':
        std::cerr << "Unknown option. Use -h for help.\n";
        return false;
      default:
        return false;
    }
  }

  return true;
}

int main(int argc, char** argv) {
  std::string config_path = "";
  std::string data_path = "";

  // Parse command line arguments
  if (!parse_arguments(argc, argv, config_path, data_path)) {
    return 1;
  }

  // Set default data path if not provided
  if (data_path.empty()) {
    auto env_data_path = std::getenv("HOLOSCAN_INPUT_PATH");
    if (env_data_path != nullptr && env_data_path[0] != '\0') {
      data_path = env_data_path;
    } else {
      data_path = std::filesystem::current_path() / "data/endoscopy";
    }
  }

  // Validate data path
  if (!std::filesystem::is_directory(data_path)) {
    std::cerr << "Data path '" << data_path
              << "' does not exist. Use --data or set HOLOSCAN_INPUT_PATH environment variable.\n";
    return 1;
  }

  // Set default config path if not provided
  if (config_path.empty()) {
    auto app_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path = app_path / "nvc_encode_writer.yaml";
  }

  // Validate config path
  if (!std::filesystem::exists(config_path)) {
    std::cerr << "Config file '" << config_path << "' does not exist.\n";
    return 1;
  }

  try {
    auto app = holoscan::make_application<NVIDIAVideoCodecApp>();
    app->set_data_path(data_path);
    app->config(config_path);

    auto& tracker = app->track();
    app->run();
    tracker.print();
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
