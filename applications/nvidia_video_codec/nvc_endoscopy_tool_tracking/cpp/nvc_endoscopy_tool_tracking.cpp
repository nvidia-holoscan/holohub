/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Include the NVIDIA Video Codec operators
#include "nv_video_decoder.hpp"
#include "nv_video_encoder.hpp"
#include "nv_video_reader.hpp"
#include "tensor_to_file.hpp"

// Include the AI operators
#include "lstm_tensor_rt_inference.hpp"
#include "tool_tracking_postprocessor.hpp"

/**
 * @brief Endoscopy Tool Tracking Application
 *
 * This application demonstrates GPU-accelerated H.264 video decoding combined with
 * LSTM-based tool tracking for endoscopic surgical videos. It reads an H.264 elementary
 * stream file, decodes it on the GPU, runs LSTM inference for tool detection and tracking,
 * and visualizes the results in real time.
 */
class EndoscopyToolTrackingApp : public holoscan::Application {
 public:
  void set_data_path(const std::string& path) { data_path_ = path; }

  void compose() override {
    using namespace holoscan;

    // Get configuration parameters
    int fps = from_config("holoviz.framerate").as<int>();
    std::string recess_period = std::to_string(fps) + "hz";

    // Validate data path
    if (!std::filesystem::exists(data_path_)) {
      throw std::runtime_error("Could not find video data: " + data_path_);
    }

    // Video dimensions
    const uint32_t width = 854;
    const uint32_t height = 480;
    const size_t source_block_size = width * height * 3 * 4;
    const size_t source_num_blocks = 2;

    // Create H264 File Reader
    auto h264_file_reader = make_operator<ops::NvVideoReaderOp>(
        "h264_file_reader",
        Arg("directory", data_path_),
        Arg("allocator", make_resource<UnboundedAllocator>("video_reader_pool")),
        make_condition<CountCondition>("count_condition", 683),  // number of frames to read
        make_condition<PeriodicCondition>("periodic_condition",
                                          Arg("recess_period") = recess_period),
        from_config("reader"));

    // Create NVIDIA Video Decoder
    auto decoder = make_operator<ops::NvVideoDecoderOp>(
        "nv_decoder",
        Arg("allocator", make_resource<UnboundedAllocator>("video_decoder_pool")),
        from_config("decoder"));

    // Create Format Converter for decoder output (NV12 -> RGB888)
    auto decoder_output_format_converter = make_operator<ops::FormatConverterOp>(
        "decoder_output_format_converter",
        Arg("pool") = make_resource<BlockMemoryPool>(
                        "pool",
                        static_cast<int32_t>(MemoryStorageType::kDevice),
                        source_block_size,
                        source_num_blocks),
        from_config("decoder_output_format_converter"));

    // Create Format Converter for RGB to float (RGB888 -> float32)
    auto rgb_float_format_converter = make_operator<ops::FormatConverterOp>(
        "rgb_float_format_converter",
        Arg("pool") = make_resource<BlockMemoryPool>(
                        "pool",
                        static_cast<int32_t>(MemoryStorageType::kDevice),
                        source_block_size,
                        source_num_blocks),
        from_config("rgb_float_format_converter"));

    // Create LSTM TensorRT Inference operator
    std::string model_file_path = data_path_ + "/tool_loc_convlstm.onnx";
    std::string engine_cache_dir = data_path_ + "/engines";

    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        Arg("model_file_path", model_file_path),
        Arg("engine_cache_dir", engine_cache_dir),
        Arg("pool", make_resource<UnboundedAllocator>("pool")),
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream",
                                                                 0,   // dev_id
                                                                 0,   // stream_flags
                                                                 0,   // stream_priority
                                                                 1,   // reserved_size
                                                                 5),  // max_size
        from_config("lstm_inference"));

    // Create Tool Tracking Postprocessor
    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        Arg("device_allocator", make_resource<UnboundedAllocator>("device_allocator")),
        from_config("tool_tracking_postprocessor"));

    // Check if output recording is enabled
    bool record_output = from_config("record_output").as<bool>();

    // Create Holoviz Visualizer
    auto visualizer = make_operator<ops::HolovizOp>(
        "visualizer",
        Arg("allocator") = make_resource<BlockMemoryPool>(
                             "allocator",
                             static_cast<int32_t>(MemoryStorageType::kDevice),
                             source_block_size,
                             source_num_blocks),
        Arg("enable_render_buffer_input") = false,
        Arg("enable_render_buffer_output") = record_output,
        from_config("holoviz"));

    // Add flows
    add_flow(h264_file_reader, decoder, {{"output", "input"}});
    add_flow(decoder, decoder_output_format_converter, {{"output", "source_video"}});
    add_flow(decoder_output_format_converter, visualizer, {{"tensor", "receivers"}});
    add_flow(decoder_output_format_converter, rgb_float_format_converter,
             {{"tensor", "source_video"}});
    add_flow(rgb_float_format_converter, lstm_inferer);
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
    add_flow(tool_tracking_postprocessor, visualizer, {{"out", "receivers"}});

    // Add recording pipeline if enabled
    if (record_output) {
      // Create Format Converter for Holoviz output
      auto holoviz_output_format_converter = make_operator<ops::FormatConverterOp>(
          "holoviz_output_format_converter",
          Arg("pool") = make_resource<BlockMemoryPool>(
                          "pool",
                          static_cast<int32_t>(MemoryStorageType::kDevice),
                          source_block_size,
                          source_num_blocks),
          from_config("holoviz_output_format_converter"));

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

      // Create Video Writer
      auto writer = make_operator<ops::TensorToFileOp>(
          "nv_writer",
          Arg("allocator", make_resource<RMMAllocator>("video_writer_allocator")),
          from_config("writer"));

      // Add recording flows
      add_flow(visualizer, holoviz_output_format_converter,
               {{"render_buffer_output", "source_video"}});
      add_flow(holoviz_output_format_converter, encoder, {{"tensor", "input"}});
      add_flow(encoder, writer, {{"output", "input"}});
    }
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

  if (data_path.empty()) {
    auto env_data_path = std::getenv("HOLOHUB_DATA_PATH");
    if (env_data_path != nullptr && env_data_path[0] != '\0') {
      data_path = env_data_path;
    } else {
      data_path = (std::filesystem::current_path() / "data/endoscopy").string();
    }
  }

  // Validate data path
  if (!std::filesystem::is_directory(data_path)) {
    std::cerr << "Data path '" << data_path
              << "' does not exist. Use --data or set HOLOHUB_DATA_PATH environment variable.\n";
    return 1;
  }

  // Set default config path if not provided
  if (config_path.empty()) {
    auto app_path = std::filesystem::path(argv[0]).parent_path();
    config_path = (app_path / "nvc_endoscopy_tool_tracking.yaml").string();
  }

  // Validate config path
  if (!std::filesystem::exists(config_path)) {
    std::cerr << "Config file '" << config_path << "' does not exist.\n";
    return 1;
  }

  try {
    auto app = holoscan::make_application<EndoscopyToolTrackingApp>();
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

