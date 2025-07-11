/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>

#include "holoscan/core/resources/gxf/gxf_component_resource.hpp"
#include "holoscan/operators/gxf_codelet/gxf_codelet.hpp"

#include "tensor_to_video_buffer.hpp"
#include "video_encoder.hpp"

// Import h.264 GXF codelets and components as Holoscan operators and resources
// Starting with Holoscan SDK v2.1.0, importing GXF codelets/components as Holoscan operators/
// resources can be done using the HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR and
// HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE macros. This new feature allows using GXF codelets
// and components in Holoscan applications without writing custom class wrappers (for C++) and
// Python wrappers (for Python) for each GXF codelet and component.
// For the VideoEncoderRequestOp class, since it needs to override the setup() to provide custom
// parameters and override the initialize() to register custom converters, it requires a custom
// class that extends the holoscan::ops::GXFCodeletOp class.

// The VideoDecoderResponseOp implements nvidia::gxf::VideoDecoderResponse and handles the output
// of the decoded H264 bit stream.
// Parameters:
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data.
// - outbuf_storage_type (uint32_t): Output Buffer Storage(memory) type used by this allocator.
//   Can be 0: kHost, 1: kDevice.
// - videodecoder_context (std::shared_ptr<holoscan::ops::VideoDecoderContext>): Decoder context
//   Handle.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoDecoderResponseOp, "nvidia::gxf::VideoDecoderResponse")

// The VideoDecoderRequestOp implements nvidia::gxf::VideoDecoderRequest and handles the input
// for the H264 bit stream decode.
// Parameters:
// - inbuf_storage_type (uint32_t): Input Buffer storage type, 0:kHost, 1:kDevice.
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition.
// - videodecoder_context (std::shared_ptr<holoscan::ops::VideoDecoderContext>): Decoder
//   context Handle.
// - codec (uint32_t): Video codec to use, 0:H264, only H264 supported. Default:0.
// - disableDPB (uint32_t): Enable low latency decode, works only for IPPP case.
// - output_format (std::string): VidOutput frame video format, nv12pl and yuv420planar are
//   supported.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoDecoderRequestOp, "nvidia::gxf::VideoDecoderRequest")

// The VideoDecoderContext implements nvidia::gxf::VideoDecoderContext and holds common variables
// and underlying context.
// Parameters:
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition required to get/set event state.
HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(VideoDecoderContext, "nvidia::gxf::VideoDecoderContext")

// The VideoReadBitstreamOp implements nvidia::gxf::VideoReadBitStream and reads h.264 video files
// from the disk at the specified input file path.
// Parameters:
// - input_file_path (std::string): Path to image file
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data
// - outbuf_storage_type (int32_t): Output Buffer storage type, 0:kHost, 1:kDevice
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoReadBitstreamOp, "nvidia::gxf::VideoReadBitStream")

// The VideoWriteBitstreamOp implements nvidia::gxf::VideoWriteBitstream and writes bit stream to
// the disk at specified output path.
// Parameters:
// - output_video_path (std::string): The file path of the output video
// - frame_width (int): The width of the output video
// - frame_height (int): The height of the output video
// - inbuf_storage_type (int): Input Buffer storage type, 0:kHost, 1:kDevice
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoWriteBitstreamOp, "nvidia::gxf::VideoWriteBitstream")

// The VideoEncoderResponseOp implements nvidia::gxf::VideoEncoderResponse and handles the output
// of the encoded YUV frames.
// Parameters:
// - pool (std::shared_ptr<Allocator>): Memory pool for allocating output data.
// - videoencoder_context (std::shared_ptr<holoscan::ops::VideoEncoderContext>): Encoder context
//   handle.
// - outbuf_storage_type (uint32_t): Output Buffer Storage(memory) type used by this allocator.
//   Can be 0: kHost, 1: kDevice. Default: 1.
HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR(VideoEncoderResponseOp, "nvidia::gxf::VideoEncoderResponse")

// The VideoEncoderContext implements nvidia::gxf::VideoEncoderContext and holds common variables
// and underlying context.
// Parameters:
// - async_scheduling_term (std::shared_ptr<holoscan::AsynchronousCondition>): Asynchronous
//   scheduling condition required to get/set event state.
HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE(VideoEncoderContext, "nvidia::gxf::VideoEncoderContext")

class App : public holoscan::Application {
 public:
  void set_datapath(const std::string& path) { datapath = path; }

  /// @brief As of Holoscan SDK 2.1.0, the extension manager must be used to register any external
  /// GXF extensions in replace of the use of YAML configuration file.
  void configure_extension() {
    auto extension_manager = executor().extension_manager();
    extension_manager->load_extension("libgxf_videodecoder.so");
    extension_manager->load_extension("libgxf_videodecoderio.so");
    extension_manager->load_extension("libgxf_videoencoder.so");
    extension_manager->load_extension("libgxf_videoencoderio.so");
  }
  void compose() override {
    using namespace holoscan;

    configure_extension();

    uint32_t width = 854;
    uint32_t height = 480;
    int64_t source_block_size = width * height * 3 * 4;
    int64_t source_num_blocks = 2;

    auto bitstream_reader = make_operator<VideoReadBitstreamOp>(
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
        make_resource<VideoDecoderContext>(Arg("async_scheduling_term") = response_condition);

    auto request_condition = make_condition<AsynchronousCondition>("request_condition");
    auto video_decoder_request =
        make_operator<VideoDecoderRequestOp>("video_decoder_request",
                                             from_config("video_decoder_request"),
                                             Arg("async_scheduling_term") = request_condition,
                                             Arg("videodecoder_context") = video_decoder_context);

    auto video_decoder_response =
        make_operator<VideoDecoderResponseOp>("video_decoder_response",
                                              from_config("video_decoder_response"),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks),
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
        Arg("device_allocator") = make_resource<UnboundedAllocator>("device_allocator"));

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
    add_flow(tool_tracking_postprocessor, visualizer, {{"out", "receivers"}});

    if (record_output) {
      auto encoder_async_condition =
          make_condition<AsynchronousCondition>("encoder_async_condition");
      auto video_encoder_context =
          make_resource<VideoEncoderContext>(Arg("scheduling_term") = encoder_async_condition);

      auto video_encoder_request = make_operator<ops::VideoEncoderRequestOp>(
          "video_encoder_request",
          from_config("video_encoder_request"),
          Arg("videoencoder_context") = video_encoder_context);

      auto video_encoder_response = make_operator<VideoEncoderResponseOp>(
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

      auto bitstream_writer = make_operator<VideoWriteBitstreamOp>(
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
