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
#include <cstdlib>
#include <filesystem>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <iostream>
#include <string>
#include <vector>

// Include the NVIDIA Video Codec operators
#include "nv_video_decoder.hpp"
#include "nv_video_encoder.hpp"

/**
 * @brief Operator to print streaming statistics
 *
 * This operator collects and prints latency statistics from the video codec pipeline.
 */
class StatsOp : public holoscan::Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StatsOp)

  StatsOp() = default;

  void setup(holoscan::OperatorSpec& spec) override { spec.input<holoscan::gxf::Entity>("input"); }

  void compute(holoscan::InputContext& op_input, holoscan::OutputContext& op_output,
               holoscan::ExecutionContext& context) override {
    auto input_message = op_input.receive<holoscan::gxf::Entity>("input");

    if (!first_frame_ignored_) {
      first_frame_ignored_ = true;
      return;
    }

    auto meta = metadata();

    encode_latency_.push_back(meta->get<double>("video_encoder_encode_latency_ms", 0.0));
    decode_latency_.push_back(meta->get<double>("video_decoder_decode_latency_ms", 0.0));
    jitter_time_.push_back(meta->get<double>("jitter_time", 0.0));
    fps_.push_back(static_cast<double>(meta->get<uint64_t>("fps", 0)));
  }

  void stop() override {
    if (!encode_latency_.empty()) {
      HOLOSCAN_LOG_INFO("Encode Latency (ms) (min, max, avg): {:.3f}, {:.3f}, {:.3f}",
                        *std::min_element(encode_latency_.begin(), encode_latency_.end()),
                        *std::max_element(encode_latency_.begin(), encode_latency_.end()),
                        std::accumulate(encode_latency_.begin(), encode_latency_.end(), 0.0) /
                            encode_latency_.size());
    }
    if (!decode_latency_.empty()) {
      HOLOSCAN_LOG_INFO("Decode Latency (ms) (min, max, avg): {:.3f}, {:.3f}, {:.3f}",
                        *std::min_element(decode_latency_.begin(), decode_latency_.end()),
                        *std::max_element(decode_latency_.begin(), decode_latency_.end()),
                        std::accumulate(decode_latency_.begin(), decode_latency_.end(), 0.0) /
                            decode_latency_.size());
    }
    if (!jitter_time_.empty()) {
      HOLOSCAN_LOG_INFO(
          "Jitter Time (ms) (min, max, avg): {:.3f}, {:.3f}, {:.3f}",
          *std::min_element(jitter_time_.begin(), jitter_time_.end()),
          *std::max_element(jitter_time_.begin(), jitter_time_.end()),
          std::accumulate(jitter_time_.begin(), jitter_time_.end(), 0.0) / jitter_time_.size());
    }
    if (!fps_.empty()) {
      HOLOSCAN_LOG_INFO("FPS (min, max, avg): {:.3f}, {:.3f}, {:.3f}",
                        *std::min_element(fps_.begin(), fps_.end()),
                        *std::max_element(fps_.begin(), fps_.end()),
                        std::accumulate(fps_.begin(), fps_.end(), 0.0) / fps_.size());
    }
  }

 private:
  bool first_frame_ignored_ = false;
  uint32_t frame_count_ = 0;

  std::vector<double> encode_latency_;
  std::vector<double> decode_latency_;
  std::vector<double> jitter_time_;
  std::vector<double> fps_;
};

/**
 * @brief NVIDIA Video Codec Encode/Decode Application
 *
 * This application demonstrates GPU-accelerated H.264/H.265 video encoding and decoding
 * using the NVIDIA Video Codec SDK and Holoscan operators. It reads video frames from a source,
 * encodes them using the NVIDIA hardware encoder, decodes the encoded stream, visualizes the
 * decoded frames in real time, and collects streaming statistics such as encode/decode latency,
 * compression ratio, and FPS.
 */
class NVIDIAVideoCodecApp : public holoscan::Application {
 public:
  void set_data_path(const std::string& path) { data_path_ = path; }

  void compose() override {
    using namespace holoscan;

    // Get configuration parameters
    uint32_t width = from_config("holoviz.width").as<uint32_t>();
    uint32_t height = from_config("holoviz.height").as<uint32_t>();
    int fps = from_config("holoviz.framerate").as<int>();
    std::string recess_period = std::to_string(fps) + "hz";
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
        make_condition<PeriodicCondition>("periodic-condition",
                                          Arg("recess_period") = recess_period),
        from_config("replayer"));

    // Create Format Converter
    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool",
                                                  static_cast<int32_t>(MemoryStorageType::kDevice),
                                                  source_block_size,
                                                  source_num_blocks),
                                              from_config("format_converter"));

    // Create NVIDIA Video Encoder
    auto encoder =
        make_operator<ops::NvVideoEncoderOp>("nv_encoder",
                                             Arg("width", width),
                                             Arg("height", height),
                                             Arg("allocator") = make_resource<BlockMemoryPool>(
                                                 "encoder_pool",
                                                 static_cast<int32_t>(MemoryStorageType::kHost),
                                                 source_block_size,
                                                 source_num_blocks),
                                             from_config("encoder"));

    // Create NVIDIA Video Decoder
    auto decoder = make_operator<ops::NvVideoDecoderOp>(
        "nv_decoder",
        Arg("allocator", make_resource<RMMAllocator>("video_decoder_allocator")),
        from_config("decoder"));

    // Create Holoviz Visualizer
    auto visualizer = make_operator<ops::HolovizOp>(
        "visualizer",
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream",
                                                                0,   // dev_id
                                                                0,   // stream_flags
                                                                0,   // stream_priority
                                                                1,   // reserved_size
                                                                5),  // max_size
        from_config("holoviz"));

    // Create Statistics Operator
    auto stats = make_operator<StatsOp>("stats");

    // Add flows
    add_flow(source, format_converter, {{"output", "source_video"}});
    add_flow(format_converter, encoder, {{"tensor", "input"}});
    add_flow(encoder, decoder, {{"output", "input"}});
    add_flow(decoder, visualizer, {{"output", "receivers"}});
    add_flow(decoder, stats, {{"output", "input"}});
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
  static struct option long_options[] = {{"help", no_argument, 0, 'h'},
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
      data_path = (std::filesystem::current_path() / "data/endoscopy").string();
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
    config_path = (app_path / "nvc_encode_decode.yaml").string();
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
