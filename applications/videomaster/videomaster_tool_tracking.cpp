/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <holoscan/std_ops.hpp>

enum class VideoSource { REPLAYER, AJA, VIDEOMASTER };

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) {
    if (source == "aja") { video_source_ = VideoSource::AJA; }
    if (source == "videomaster") { video_source_ = VideoSource::VIDEOMASTER; }
  }
  enum class Record { NONE, INPUT, VISUALIZER };
  void set_record(const std::string& record) {
    if (record == "input") {
      record_type_ = Record::INPUT;
    } else if (record == "visualizer") {
      record_type_ = Record::VISUALIZER;
    }
  }

  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source;
    std::shared_ptr<Operator> recorder;
    std::shared_ptr<Operator> recorder_format_converter;

    const bool is_video_source =
        video_source_ == VideoSource::AJA || video_source_ == VideoSource::VIDEOMASTER;
    const bool is_aja_overlay_enabled =
        video_source_ == VideoSource::AJA && from_config("aja.enable_overlay").as<bool>();
    uint32_t width = 0;
    uint32_t height = 0;
    uint64_t source_block_size = 0;
    uint64_t source_num_blocks = 0;

    switch (video_source_) {
      case VideoSource::AJA:
        width = from_config("aja.width").as<uint32_t>();
        height = from_config("aja.height").as<uint32_t>();
        source = make_operator<ops::AJASourceOp>("aja", from_config("aja"));
        source_block_size = width * height * 4 * 4;
        source_num_blocks = from_config("aja.rdma").as<bool>() ? 3 : 4;
        break;
      case VideoSource::VIDEOMASTER:
        width = from_config("videomaster.width").as<uint32_t>();
        height = from_config("videomaster.height").as<uint32_t>();
        source = make_operator<ops::VideoMasterSourceOp>(
            "videomaster",
            from_config("videomaster"),
            Arg("pool") = make_resource<UnboundedAllocator>("pool"));
        source_block_size = width * height * 4 * 4;
        source_num_blocks = from_config("videomaster.rdma").as<bool>() ? 3 : 4;
        break;
      default:
        width = 854;
        height = 480;
        source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"));
        source_block_size = width * height * 3 * 4;
        source_num_blocks = 2;
        break;
    }

    if (record_type_ != Record::NONE) {
      if (((record_type_ == Record::INPUT) && is_video_source) ||
          (record_type_ == Record::VISUALIZER)) {
        recorder_format_converter = make_operator<ops::FormatConverterOp>(
            "recorder_format_converter",
            from_config("recorder_format_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
      }
      recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));
    }

    std::string formater_converter_config_name;
    switch (video_source_) {
      case VideoSource::AJA:
        formater_converter_config_name = "format_converter_aja";
        break;
      case VideoSource::VIDEOMASTER:
        formater_converter_config_name = "format_converter_videomaster";
        break;
      default:
        formater_converter_config_name = "format_converter_replayer";
        break;
    }
    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              from_config(formater_converter_config_name),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks));

    auto lstm_inferer = make_operator<ops::LSTMTensorRTInferenceOp>(
        "lstm_inferer",
        from_config("lstm_inference"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));

    // A known issue using the BlockMemoryPool when converting the ONNX model to TRT from
    // the LSTMTensorRTInferenceOp causes the inference to produce wrong values the first
    // time the conversion is done inline.
    // Switching from BlockMemoryPool to UnboundedAllocator or increasing the block size of the
    // BlockMemoryPool seems to fix the issue.
    // Using TensorRT 8.4.1 and above seems also to be fixing the issue.
    // const uint64_t tool_tracking_postprocessor_block_size = 107 * 60 * 7 * 4;
    // const uint64_t tool_tracking_postprocessor_num_blocks = 2;
    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        from_config("tool_tracking_postprocessor"),
        //  Arg("device_allocator") = make_resource<BlockMemoryPool>("device_allocator",
        //                                   1,
        //                                   tool_tracking_postprocessor_block_size,
        //                                   tool_tracking_postprocessor_num_blocks),
        Arg("device_allocator") = make_resource<UnboundedAllocator>("device_allocator"),
        Arg("host_allocator") = make_resource<UnboundedAllocator>("host_allocator"));

    std::shared_ptr<BlockMemoryPool> visualizer_allocator;
    if ((record_type_ == Record::VISUALIZER) && !is_video_source) {
      visualizer_allocator =
          make_resource<BlockMemoryPool>("allocator", 1, source_block_size, source_num_blocks);
    }
    std::shared_ptr<ops::HolovizOp> visualizer = make_operator<ops::HolovizOp>(
        "holoviz",
        from_config(is_aja_overlay_enabled ? "holoviz_overlay" : "holoviz"),
        Arg("width") = width,
        Arg("height") = height,
        Arg("enable_render_buffer_input") = is_aja_overlay_enabled,
        Arg("enable_render_buffer_output") = is_aja_overlay_enabled ||
                                             (record_type_ == Record::VISUALIZER),
        Arg("allocator") = visualizer_allocator);

    // Flow definition
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
    add_flow(tool_tracking_postprocessor, visualizer, {{"out", "receivers"}});

    switch (video_source_) {
      case VideoSource::AJA:
        add_flow(source, format_converter, {{"video_buffer_output", "source_video"}});
        break;
      case VideoSource::VIDEOMASTER:
        add_flow(source, format_converter, {{"signal", "source_video"}});
        break;
      default:
        add_flow(source, format_converter, {{"output", "source_video"}});
        break;
    }
    add_flow(format_converter, lstm_inferer);

    if (is_aja_overlay_enabled) {
      // Overlay buffer flow between AJA source and visualizer
      add_flow(source, visualizer, {{"overlay_buffer_output", "render_buffer_input"}});
      add_flow(visualizer, source, {{"render_buffer_output", "overlay_buffer_input"}});
    } else {
      switch (video_source_) {
        case VideoSource::AJA:
          add_flow(source, visualizer, {{"video_buffer_output", "receivers"}});
          break;
        case VideoSource::VIDEOMASTER:
          add_flow(source, visualizer, {{"signal", "receivers"}});
          break;
        default:
          add_flow(source, visualizer, {{"output", "receivers"}});
          break;
      }
    }

    if (record_type_ == Record::INPUT) {
      switch (video_source_) {
        case VideoSource::AJA:
          add_flow(source, recorder_format_converter, {{"video_buffer_output", "source_video"}});
          add_flow(recorder_format_converter, recorder);
          break;
        case VideoSource::VIDEOMASTER:
          add_flow(source, recorder_format_converter, {{"signal", "source_video"}});
          add_flow(recorder_format_converter, recorder);
          break;
        default:
          add_flow(source, recorder);
          break;
      }
    } else if (record_type_ == Record::VISUALIZER) {
      add_flow(visualizer, recorder_format_converter, {{"render_buffer_output", "source_video"}});
      add_flow(recorder_format_converter, recorder);
    }
  }

 private:
  VideoSource video_source_ = VideoSource::REPLAYER;
  Record record_type_ = Record::NONE;
};

int main(int argc, char** argv) {
  holoscan::load_env_log_level();

  auto app = holoscan::make_application<App>();

  if (argc == 2) {
    app->config(argv[1]);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/app_config.yaml";
    app->config(config_path);
  }
  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);
  auto record_type = app->from_config("record_type").as<std::string>();
  app->set_record(record_type);

  app->run();

  return 0;
}
