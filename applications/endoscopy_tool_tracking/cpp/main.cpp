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

#include "holoscan/holoscan.hpp"
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_recorder/video_stream_recorder.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <lstm_tensor_rt_inference.hpp>
#include <tool_tracking_postprocessor.hpp>
#ifdef VTK_RENDERER
#include <vtk_renderer.hpp>
#endif

#ifdef AJA_SOURCE
#include <aja_source.hpp>
#endif

#ifdef DELTACAST_VIDEOMASTER
#include <videomaster_source.hpp>
#include <videomaster_transmitter.hpp>
#endif

#ifdef YUAN_QCAP
#include <qcap_source.hpp>
#endif

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) { source_ = source; }
  void set_visualizer_name(const std::string& visualizer_name) {
    this->visualizer_name = visualizer_name;
  }

  enum class Record { NONE, INPUT, VISUALIZER };

  void set_record(const std::string& record) {
    if (record == "input") {
      record_type_ = Record::INPUT;
    } else if (record == "visualizer") {
      record_type_ = Record::VISUALIZER;
    }
  }

  void set_datapath(const std::string& path) { datapath = path; }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;
    std::shared_ptr<Operator> recorder;
    std::shared_ptr<Operator> recorder_format_converter;
    std::shared_ptr<Operator> visualizer_operator;

    const bool use_rdma = from_config("external_source.rdma").as<bool>();
    const bool overlay_enabled = (source_ != "replayer") && (this->visualizer_name == "holoviz") &&
                                 from_config("external_source.enable_overlay").as<bool>();

    const std::string input_video_signal =
        this->visualizer_name == "holoviz" ? "receivers" : "videostream";
    const std::string input_annotations_signal =
        this->visualizer_name == "holoviz" ? "receivers" : "annotations";

    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t source_block_size = 0;
    uint64_t source_num_blocks = 0;

    if (source_ == "aja") {
      width = from_config("aja.width").as<uint32_t>();
      height = from_config("aja.height").as<uint32_t>();
      source = make_operator<ops::AJASourceOp>(
          "aja", from_config("aja"), from_config("external_source"));
      source_block_size = width * height * 4 * 4;
      source_num_blocks = use_rdma ? 3 : 4;
    } else if (source_ == "yuan") {
      width = from_config("yuan.width").as<uint32_t>();
      height = from_config("yuan.height").as<uint32_t>();
#ifdef YUAN_QCAP
      source = make_operator<ops::QCAPSourceOp>("yuan", from_config("yuan"));
#endif
      source_block_size = width * height * 4 * 4;
      source_num_blocks = use_rdma ? 3 : 4;
    } else if (source_ == "deltacast") {
      width = from_config("deltacast.width").as<uint32_t>();
      height = from_config("deltacast.height").as<uint32_t>();
#ifdef DELTACAST_VIDEOMASTER
      source = make_operator<ops::VideoMasterSourceOp>(
          "deltacast",
          from_config("deltacast"),
          from_config("external_source"),
          Arg("pool") = make_resource<UnboundedAllocator>("pool"));
#endif
      source_block_size = width * height * 4 * 4;
      source_num_blocks = use_rdma ? 3 : 4;
    } else {  // Replayer
      width = 854;
      height = 480;
      source = make_operator<ops::VideoStreamReplayerOp>(
          "replayer", from_config("replayer"), Arg("directory", datapath));
      source_block_size = width * height * 3 * 4;
      source_num_blocks = 2;
    }

    if (record_type_ != Record::NONE) {
      if (((record_type_ == Record::INPUT) && (source_ != "replayer")) ||
          (record_type_ == Record::VISUALIZER)) {
        recorder_format_converter = make_operator<ops::FormatConverterOp>(
            "recorder_format_converter",
            from_config("recorder_format_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
      }
      recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));
    }

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              from_config("format_converter_" + source_),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks),
                                              Arg("cuda_stream_pool") = cuda_stream_pool);

    const std::string model_file_path = datapath + "/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = datapath + "/engines";

    const uint64_t lstm_inferer_block_size = 107 * 60 * 7 * 4;
    const uint64_t lstm_inferer_num_blocks = 2 + 5 * 2;
    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("model_file_path", model_file_path),
        Arg("engine_cache_dir", engine_cache_dir),
        Arg("pool") = make_resource<BlockMemoryPool>(
            "pool", 1, lstm_inferer_block_size, lstm_inferer_num_blocks),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    const uint64_t tool_tracking_postprocessor_block_size = 107 * 60 * 7 * 4;
    const uint64_t tool_tracking_postprocessor_num_blocks = 2;
    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        Arg("device_allocator") =
            make_resource<BlockMemoryPool>("device_allocator",
                                           1,
                                           tool_tracking_postprocessor_block_size,
                                           tool_tracking_postprocessor_num_blocks),
        Arg("host_allocator") = make_resource<UnboundedAllocator>("host_allocator"));

    if (this->visualizer_name == "holoviz") {
      std::shared_ptr<BlockMemoryPool> visualizer_allocator;
      if ((record_type_ == Record::VISUALIZER) && source_ == "replayer") {
        visualizer_allocator =
            make_resource<BlockMemoryPool>("allocator", 1, source_block_size, source_num_blocks);
      }
      visualizer_operator = make_operator<ops::HolovizOp>(
          "holoviz",
          from_config(overlay_enabled ? "holoviz_overlay" : "holoviz"),
          Arg("width") = width,
          Arg("height") = height,
          Arg("enable_render_buffer_input") = overlay_enabled,
          Arg("enable_render_buffer_output") =
              overlay_enabled || (record_type_ == Record::VISUALIZER),
          Arg("allocator") = visualizer_allocator,
          Arg("cuda_stream_pool") = cuda_stream_pool);
    }
#ifdef VTK_RENDERER
    if (this->visualizer_name == "vtk") {
      visualizer_operator = make_operator<ops::VtkRendererOp>("vtk",
                                                      from_config("vtk_op"),
                                                      Arg("width") = width,
                                                      Arg("height") = height);
    }
#endif

    // Flow definition
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});

    if (this->visualizer_name == "holoviz") {
      add_flow(tool_tracking_postprocessor,
               visualizer_operator,
               {{"out_coords", input_annotations_signal}, {"out_mask", input_annotations_signal}});
    } else {
      // device tensor on the out_mask port is not used by VtkRendererOp
      add_flow(tool_tracking_postprocessor,
               visualizer_operator,
               {{"out_coords", input_annotations_signal}});
    }

    std::string output_signal = "output";  // replayer output signal name
    if (source_ == "deltacast") {
      output_signal = "signal";
    } else if (source_ == "aja" || source_ == "yuan") {
      output_signal = "video_buffer_output";
    }

    add_flow(source, format_converter, {{output_signal, "source_video"}});

    add_flow(format_converter, lstm_inferer);

    if (source_ == "deltacast") {
#ifdef DELTACAST_VIDEOMASTER
      if (overlay_enabled) {
        // Overlay buffer flow between source and visualizer
        auto overlayer = make_operator<ops::VideoMasterTransmitterOp>(
            "videomaster_overlayer",
            from_config("videomaster"),
            Arg("pool") = make_resource<UnboundedAllocator>("pool"));
        auto overlay_format_converter_videomaster = make_operator<ops::FormatConverterOp>(
            "overlay_format_converter",
            from_config("deltacast_overlay_format_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
        add_flow(visualizer_operator,
                 overlay_format_converter_videomaster,
                 {{"render_buffer_output", ""}});
        add_flow(overlay_format_converter_videomaster, overlayer);
      } else {
        auto visualizer_format_converter_videomaster = make_operator<ops::FormatConverterOp>(
            "visualizer_format_converter",
            from_config("deltacast_visualizer_format_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
        auto drop_alpha_channel_converter = make_operator<ops::FormatConverterOp>(
            "drop_alpha_channel_converter",
            from_config("deltacast_drop_alpha_channel_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
        add_flow(source, drop_alpha_channel_converter);
        add_flow(drop_alpha_channel_converter, visualizer_format_converter_videomaster);
        add_flow(visualizer_format_converter_videomaster, visualizer_operator, {{"", "receivers"}});
      }
#endif
    } else {
      if (overlay_enabled) {
        // Overlay buffer flow between source and visualizer_operator
        add_flow(source, visualizer_operator, {{"overlay_buffer_output", "render_buffer_input"}});
        add_flow(visualizer_operator, source, {{"render_buffer_output", "overlay_buffer_input"}});
      } else {
        add_flow(source, visualizer_operator, {{output_signal, input_video_signal}});
      }
    }

    if (record_type_ == Record::INPUT) {
      if (source_ != "replayer") {
        add_flow(source, recorder_format_converter, {{output_signal, "source_video"}});
        add_flow(recorder_format_converter, recorder);
      } else {
        add_flow(source, recorder);
      }
    } else if (record_type_ == Record::VISUALIZER && this->visualizer_name == "holoviz") {
      add_flow(visualizer_operator,
               recorder_format_converter,
               {{"render_buffer_output", "source_video"}});
      add_flow(recorder_format_converter, recorder);
    }
  }

 private:
  std::string source_ = "replayer";
  std::string visualizer_name = "holoviz";
  Record record_type_ = Record::NONE;
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

/** Main function */
int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return 1; }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/endoscopy_tool_tracking.yaml";
    app->config(config_path);
  }

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);

  auto record_type = app->from_config("record_type").as<std::string>();
  app->set_record(record_type);

  auto visualizer_name = app->from_config("visualizer").as<std::string>();
  app->set_visualizer_name(visualizer_name);

  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
