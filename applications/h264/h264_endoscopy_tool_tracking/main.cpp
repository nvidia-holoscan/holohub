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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "tensor_to_video_buffer.hpp"
#include "video_decoder.hpp"
#include "video_encoder.hpp"
#include "video_read_bitstream.hpp"
#include "video_write_bitstream.hpp"

class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath = path; }

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
            make_resource<BlockMemoryPool>("pool", 0, source_block_size, source_num_blocks));

    auto response_condition = make_condition<AsynchronousCondition>("response_condition");
    auto video_decoder_context =
        make_resource<ops::VideoDecoderContext>(Arg("async_scheduling_term") = response_condition);

    auto request_condition = make_condition<AsynchronousCondition>("request_condition");
    auto video_decoder_request = make_operator<ops::VideoDecoderRequestOp>(
        "video_decoder_request",
        from_config("video_decoder_request"),
        Arg("async_scheduling_term") = request_condition,
        Arg("videodecoder_context") = video_decoder_context);

    auto video_decoder_response = make_operator<ops::VideoDecoderResponseOp>(
        "video_decoder_response",
        from_config("video_decoder_response"),
        Arg("pool") =
            make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks),
        Arg("videodecoder_context") = video_decoder_context);

    auto decoder_output_format_converter =
        make_operator<ops::FormatConverterOp>("decoder_output_format_converter",
                                              from_config("decoder_output_format_converter"),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks));

    auto rgb_float_format_converter =
        make_operator<ops::FormatConverterOp>("rgb_float_format_converter",
                                              from_config("rgb_float_format_converter"),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks));

    const std::string model_file_path = datapath + "/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = datapath + "/engines";

    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("model_file_path", model_file_path),
        Arg("engine_cache_dir", engine_cache_dir),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));

    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        from_config("tool_tracking_postprocessor"),
        Arg("device_allocator") = make_resource<UnboundedAllocator>("device_allocator"),
        Arg("host_allocator") = make_resource<UnboundedAllocator>("host_allocator"));

    const bool record_output = from_config("record_output").as<bool>();

    std::shared_ptr<BlockMemoryPool> visualizer_allocator =
        make_resource<BlockMemoryPool>("allocator", 1, source_block_size, source_num_blocks);
    auto visualizer =
        make_operator<ops::HolovizOp>("holoviz",
                                      from_config("holoviz"),
                                      Arg("width") = width,
                                      Arg("height") = height,
                                      Arg("enable_render_buffer_input") = false,
                                      Arg("enable_render_buffer_output") = record_output == true,
                                      Arg("allocator") = visualizer_allocator);

    add_flow(bitstream_reader, video_decoder_request, {{"output_transmitter", "input_frame"}});
    add_flow(video_decoder_response,
             decoder_output_format_converter,
             {{"output_transmitter", "source_video"}});
    add_flow(decoder_output_format_converter, visualizer, {{"tensor", "receivers"}});
    add_flow(
        decoder_output_format_converter, rgb_float_format_converter, {{"tensor", "source_video"}});
    add_flow(rgb_float_format_converter, lstm_inferer);
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
    add_flow(tool_tracking_postprocessor,
             visualizer,
             {{"out_coords", "receivers"}, {"out_mask", "receivers"}});

    if (record_output) {
      auto encoder_async_condition =
          make_condition<AsynchronousCondition>("encoder_async_condition");
      auto video_encoder_context =
          make_resource<ops::VideoEncoderContext>(Arg("scheduling_term") = encoder_async_condition);

      auto video_encoder_request = make_operator<ops::VideoEncoderRequestOp>(
          "video_encoder_request",
          from_config("video_encoder_request"),
          Arg("videoencoder_context") = video_encoder_context);

      auto video_encoder_response = make_operator<ops::VideoEncoderResponseOp>(
          "video_encoder_response",
          from_config("video_encoder_response"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks),
          Arg("videoencoder_context") = video_encoder_context);

      auto holoviz_output_format_converter = make_operator<ops::FormatConverterOp>(
          "holoviz_output_format_converter",
          from_config("holoviz_output_format_converter"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));

      auto encoder_input_format_converter = make_operator<ops::FormatConverterOp>(
          "encoder_input_format_converter",
          from_config("encoder_input_format_converter"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));

      auto tensor_to_video_buffer = make_operator<ops::TensorToVideoBufferOp>(
          "tensor_to_video_buffer", from_config("tensor_to_video_buffer"));

      auto bitstream_writer = make_operator<ops::VideoWriteBitstreamOp>(
          "bitstream_writer",
          from_config("bitstream_writer"),
          Arg("output_video_path", datapath + "/surgical_video_output.264"),
          Arg("input_crc_file_path", datapath + "/surgical_video_output.txt"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 0, source_block_size, source_num_blocks));

      add_flow(
          visualizer, holoviz_output_format_converter, {{"render_buffer_output", "source_video"}});
      add_flow(holoviz_output_format_converter,
               encoder_input_format_converter,
               {{"tensor", "source_video"}});
      add_flow(encoder_input_format_converter, tensor_to_video_buffer, {{"tensor", "in_tensor"}});
      add_flow(
          tensor_to_video_buffer, video_encoder_request, {{"out_video_buffer", "input_frame"}});
      add_flow(video_encoder_response, bitstream_writer, {{"output_transmitter", "data_receiver"}});
    }
  }

 private:
  std::string datapath = "data/endoscopy";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {{"data", required_argument, 0, 'd'}, {0, 0, 0, 0}};

  while (int c = getopt_long(argc, argv, "d", long_options, NULL)) {
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

  if (optind < argc) { config_name = argv[optind++]; }
  return true;
}

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();
  GxfSetSeverity(app->executor().context(), GXF_SEVERITY_WARNING);

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return 1; }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/h264_endoscopy_tool_tracking.yaml";
    app->config(config_path);
  }

  if (data_path != "") app->set_datapath(data_path);
  app->run();

  return 0;
}
