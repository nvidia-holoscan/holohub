/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include "holoscan/core/resources/gxf/gxf_component_resource.hpp"
#include "holoscan/operators/gxf_codelet/gxf_codelet.hpp"

#include "dds_video_publisher.hpp"
#include "dds_video_subscriber.hpp"

#include "append_timestamp.hpp"
#include "tensor_to_video_buffer.hpp"
#include "video_encoder.hpp"

#include <getopt.h>
#include <filesystem>

// Import h.264 GXF codelets and components as Holoscan operators and resources
// Starting with Holoscan SDK v2.1.0, importing GXF codelets/components as Holoscan operators/
// resources can be done using the HOLOSCAN_WRAP_GXF_CODELET_AS_OPERATOR and
// HOLOSCAN_WRAP_GXF_COMPONENT_AS_RESOURCE macros. This new feature allows using GXF codelets
// and components in Holoscan applications without writing custom class wrappers (for C++) and
// Python wrappers (for Python) for each GXF codelet and component.
// For the VideoEncoderRequestOp class, since it needs to override the setup() to provide custom
// parameters and override the initialize() to register custom converters, it requires a custom
// class that extends the holoscan::ops::GXFCodeletOp class.

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

/**
 * @brief Application to publish a V4L2 video stream to DDS.
 */
class StreamingServer : public holoscan::Application {
 public:
  explicit StreamingServer(std::string video_path) : video_path_(video_path) {}

  void configure_extension() {
    auto extension_manager = executor().extension_manager();
    extension_manager->load_extension("libgxf_videoencoder.so");
    extension_manager->load_extension("libgxf_videoencoderio.so");
  }

  void compose() override {
    using namespace holoscan;

    configure_extension();

    uint32_t width = 854;
    uint32_t height = 480;
    uint32_t source_block_size = width * height * 3 * 4;
    uint32_t source_num_blocks = 2;

    auto source = from_config("source").as<std::string>();

    std::shared_ptr<Operator> source_operator;
    std::shared_ptr<Operator> format_converter_rgba8888;
    std::shared_ptr<Operator> format_converter_rgb888;

    if (source == "replayer") {
      HOLOSCAN_LOG_INFO("Using video path: {}", video_path_);
      source_operator = make_operator<ops::VideoStreamReplayerOp>(
          "replayer",
          Arg("allocator", make_resource<RMMAllocator>("video_replayer_allocator")),
          from_config("replayer"),
          Arg("directory", video_path_));

      HOLOSCAN_LOG_INFO("Using format converter");
      format_converter_rgb888 = make_operator<ops::FormatConverterOp>(
          "format_converter",
          from_config("format_converter_rgb888"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
    } else if (source == "v4l2") {
      HOLOSCAN_LOG_INFO("Using v4l2");
      width = from_config("v4l2.width").as<int>();
      height = from_config("v4l2.height").as<int>();
      source_block_size = width * height * 3 * 4;
      source_num_blocks = 2;
      source_operator = make_operator<ops::V4L2VideoCaptureOp>(
          "v4l2",
          from_config("v4l2"),
          Arg("allocator") = make_resource<UnboundedAllocator>("pool"));

      format_converter_rgba8888 = make_operator<ops::FormatConverterOp>(
          "format_converter_rgba8888",
          from_config("format_converter_rgba8888"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));

      format_converter_rgb888 = make_operator<ops::FormatConverterOp>(
          "format_converter_rgb888",
          from_config("format_converter_rgb888"),
          Arg("pool") =
              make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
    } else {
      HOLOSCAN_LOG_ERROR("Invalid source: {}", source);
      throw std::runtime_error("Invalid source configuration: " + source);
    }
    HOLOSCAN_LOG_INFO("Video width: {}, height: {}", width, height);

    auto video_replayer = make_operator<ops::VideoStreamReplayerOp>(
        "replayer",
        Arg("allocator", make_resource<RMMAllocator>("video_replayer_allocator")),
        from_config("replayer"),
        Arg("directory", video_path_));

    auto tensor_to_video_buffer = make_operator<ops::TensorToVideoBufferOp>(
        "tensor_to_video_buffer", from_config("tensor_to_video_buffer"));

    auto encoder_async_condition = make_condition<AsynchronousCondition>("encoder_async_condition");
    auto video_encoder_context =
        make_resource<VideoEncoderContext>(Arg("scheduling_term") = encoder_async_condition);

    auto video_encoder_request = make_operator<ops::VideoEncoderRequestOp>(
        "video_encoder_request",
        Arg("input_width", width),
        Arg("input_height", height),
        from_config("video_encoder_request"),
        Arg("videoencoder_context") = video_encoder_context);

    auto video_encoder_response =
        make_operator<VideoEncoderResponseOp>("video_encoder_response",
                                              from_config("video_encoder_response"),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks),
                                              Arg("videoencoder_context") = video_encoder_context);
    auto video_publisher = make_operator<ops::DDSVideoPublisherOp>(
        "video_publisher",
        Arg("width", width),
        Arg("height", height),
        from_config("video_publisher"));

    auto holoviz = make_operator<ops::HolovizOp>("holoviz",
                                                 Arg("window_title") = "DDS Publisher",
                                                 Arg("width", width),
                                                 Arg("height", height),
                                                 from_config("holoviz"));

    if (source == "replayer") {
      add_flow(source_operator, format_converter_rgb888, {{"output", "source_video"}});
      add_flow(source_operator, holoviz, {{"output", "receivers"}});
    } else if (source == "v4l2") {
      add_flow(source_operator, format_converter_rgba8888, {{"signal", "source_video"}});
      add_flow(format_converter_rgba8888, format_converter_rgb888, {{"tensor", "source_video"}});
      add_flow(format_converter_rgba8888, holoviz, {{"tensor", "receivers"}});
    }
    add_flow(format_converter_rgb888, tensor_to_video_buffer, {{"tensor", "in_tensor"}});
    add_flow(tensor_to_video_buffer, video_encoder_request, {{"out_video_buffer", "input_frame"}});
    add_flow(video_encoder_response, video_publisher, {{"output_transmitter", "input"}});
  }

 private:
  uint32_t domain_id_;
  uint32_t stream_id_;
  std::string video_path_;
};

/**
 * @brief Application to render a DDS video stream (published by the DDSVideoPublisher)
 * and shapes (published by the RTI Connext Shapes Demo) to Holoviz.
 */
class StreamingClient : public holoscan::Application {
 public:
  explicit StreamingClient() {}

  void configure_extension() {
    auto extension_manager = executor().extension_manager();
    extension_manager->load_extension("libgxf_videodecoder.so");
    extension_manager->load_extension("libgxf_videodecoderio.so");
  }

  void compose() override {
    using namespace holoscan;

    configure_extension();

    uint32_t width = 854;
    uint32_t height = 480;
    uint32_t source_block_size = width * height * 3 * 4;
    uint32_t source_num_blocks = 2;

    if (from_config("source").as<std::string>() == "v4l2") {
      width = from_config("v4l2.width").as<int>();
      height = from_config("v4l2.height").as<int>();
    }

    HOLOSCAN_LOG_INFO("Video width: {}, height: {}", width, height);

    std::shared_ptr<UnboundedAllocator> allocator = make_resource<UnboundedAllocator>("pool");

    //  DDS Video Subscriber
    auto video_subscriber = make_operator<ops::DDSVideoSubscriberOp>(
        "video_subscriber",
        Arg("allocator", allocator),
        from_config("video_subscriber"));

    auto append_timestamp = make_operator<ops::AppendTimestampOp>("append_timestamp");

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

    // Holoviz (initialize with the default input spec for the video stream)
    auto holoviz = make_operator<ops::HolovizOp>("holoviz",
                                                 Arg("window_title") = "DDS Subscriber",
                                                 Arg("width", width),
                                                 Arg("height", height),
                                                 from_config("holoviz"));

    add_flow(video_subscriber, append_timestamp, {{"output", "in_tensor"}});
    add_flow(append_timestamp, video_decoder_request, {{"out_tensor", "input_frame"}});
    add_flow(video_decoder_response, holoviz, {{"output_transmitter", "receivers"}});
  }
};

void usage() {
  std::cout << "Usage: dds_video {-p | -s} [options]" << std::endl
            << std::endl
            << "Options" << std::endl
            << "  -p,    --publisher    Run as a publisher" << std::endl
            << "  -s,    --subscriber   Run as a subscriber" << std::endl
            << "  -v VIDEO_PATH, --video=VIDEO_PATH        Use the specified video path"
            << std::endl
            << "  -c CONFIG_PATH, --config=CONFIG_PATH        Use the specified config path"
            << std::endl;
}

int main(int argc, char** argv) {
  bool publisher = false;
  bool subscriber = false;
  std::string video_path = "";
  std::string config_path = "";
  struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                  {"publisher", no_argument, 0, 'p'},
                                  {"subscriber", no_argument, 0, 's'},
                                  {"video", required_argument, 0, 'v'},
                                  {"config", optional_argument, 0, 'c'},
                                  {0, 0, 0, 0}};

  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "hpsi:d:v:c::", long_options, &option_index);
    if (c == -1) {
      break;
    }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        usage();
        return 0;
      case 'p':
        publisher = true;
        break;
      case 's':
        subscriber = true;
        break;
      case 'v':
        video_path = argument;
        break;
      case 'c':
        config_path = argument;
        break;
      default:
        HOLOSCAN_LOG_ERROR("Unhandled option '{}'", static_cast<char>(c));
    }
  }

  if (!publisher && !subscriber) {
    HOLOSCAN_LOG_ERROR("Must provide either -p or -s for publisher or subscriber, respectively");
    usage();
    return -1;
  }

  if (publisher && video_path.empty()) {
    HOLOSCAN_LOG_ERROR("Video path is required when running as publisher");
    usage();
    return -1;
  }

  if (config_path.empty()) {
    auto exe_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path = exe_path / "dds_h264.yaml";
    HOLOSCAN_LOG_INFO("No config path provided, using default config: {}", config_path);
  } else {
    auto canonical_path = std::filesystem::canonical(config_path);
    config_path = canonical_path.string();
    HOLOSCAN_LOG_INFO("Using config path: {}", config_path);
  }

  HOLOSCAN_LOG_INFO("Starting {}...", publisher ? "publisher" : "subscriber");

  if (publisher) {
    auto app = holoscan::make_application<StreamingServer>(video_path);
    app->config(config_path);
    app->run();
  } else if (subscriber) {
    auto app = holoscan::make_application<StreamingClient>();
    app->config(config_path);
    app->run();
  }

  return 0;
}
