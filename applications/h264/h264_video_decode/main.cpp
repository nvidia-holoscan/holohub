/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gxf/core/gxf.h>
#include <gxf/core/gxf_ext.h>
#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "video_decoder.hpp"
#include "video_read_bitstream.hpp"

class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) {
      datapath = path;
  }

  void compose() override {
    using namespace holoscan;

    uint32_t width = 854;
    uint32_t height = 480;
    int64_t source_block_size = width * height * 3 * 4;
    int64_t source_num_blocks = 2;

    auto bitstream_reader = make_operator<ops::VideoReadBitstreamOp>(
        "bitstream_reader",
        from_config("bitstream_reader"),
        Arg("input_file_path", datapath + "/surgical_video.264"),
        make_condition<CountCondition>(750),
        make_condition<PeriodicCondition>("periodic-condition",
                                          Arg("recess_period") = std::string("25hz")),
        Arg("pool") =
            make_resource<BlockMemoryPool>(
                "pool", 0, source_block_size, source_num_blocks));

    auto response_condition =
        make_condition<AsynchronousCondition>("response_condition");
    auto video_decoder_context = make_resource<ops::VideoDecoderContext>(
        "decoder-context", Arg("async_scheduling_term") = response_condition);

    auto request_condition =
        make_condition<AsynchronousCondition>("request_condition");
    auto video_decoder_request = make_operator<ops::VideoDecoderRequestOp>(
        "video_decoder_request",
        from_config("video_decoder_request"),
        Arg("async_scheduling_term") = request_condition,
        Arg("videodecoder_context") = video_decoder_context);

    auto video_decoder_response = make_operator<ops::VideoDecoderResponseOp>(
        "video_decoder_response",
        from_config("video_decoder_response"),
        Arg("pool") =
            make_resource<BlockMemoryPool>(
                "pool", 1, source_block_size, source_num_blocks),
        Arg("videodecoder_context") = video_decoder_context);

    auto decoder_output_format_converter =
        make_operator<ops::FormatConverterOp>("decoder_output_format_converter",
            from_config("decoder_output_format_converter"),
            Arg("pool") = make_resource<BlockMemoryPool>(
                "pool", 1, source_block_size, source_num_blocks));

    std::shared_ptr<BlockMemoryPool> visualizer_allocator =
        make_resource<BlockMemoryPool>(
            "allocator", 1, source_block_size, source_num_blocks);
    auto visualizer = make_operator<ops::HolovizOp>("holoviz",
        from_config("holoviz"),
        Arg("width") = width,
        Arg("height") = height,
        Arg("enable_render_buffer_input") = false,
        Arg("enable_render_buffer_output") = false,
        Arg("allocator") = visualizer_allocator);

    add_flow(bitstream_reader, video_decoder_request,
        {{"output_transmitter", "input_frame"}});
    add_flow(video_decoder_response, decoder_output_format_converter,
        {{"output_transmitter", "source_video"}});
    add_flow(decoder_output_format_converter, visualizer,
        {{"tensor", "receivers"}});
  }

 private:
  std::string datapath = "data/endoscopy";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name,
    std::string& data_path) {
  static struct option long_options[] = {
      {"data",    required_argument, 0,  'd' },
      {0,         0,                 0,  0 }
  };

  while (int c = getopt_long(argc, argv, "d",
                   long_options, NULL))  {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) {
    config_name = argv[optind++];
  }
  return true;
}

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();
  GxfSetSeverity(app->executor().context(), GXF_SEVERITY_WARNING);

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) {
    return 1;
  }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/h264_video_decode.yaml";
    app->config(config_path);
  }

  if (data_path != "") app->set_datapath(data_path);
  app->run();

  return 0;
}
