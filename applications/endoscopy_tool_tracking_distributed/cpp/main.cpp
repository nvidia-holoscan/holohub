/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
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
#include <holoscan/operators/aja_source/aja_source.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <iostream>
#include <lstm_tensor_rt_inference.hpp>
#include <string>
#include <tool_tracking_postprocessor.hpp>
#include "holoscan/holoscan.hpp"

#ifdef YUAN_QCAP
#include <qcap_source.hpp>
#endif

using namespace holoscan;
class VideInputFragment : public holoscan::Fragment {
 private:
  bool use_rdma;
  std::shared_ptr<holoscan::Operator> input_op_;
  std::string input_dir_;
  std::string source_;
  uint32_t height_ = 0;
  uint32_t width_ = 0;
  uint64_t source_block_size_ = 0;
  uint64_t source_num_blocks_ = 0;

 public:
  std::string AnyPrint(const std::any& value) {
    std::cout << size_t(&value) << ", " << value.type().name() << " ";
    if (auto x = std::any_cast<int>(&value)) { return "int(" + std::to_string(*x) + ")"; }
    if (auto x = std::any_cast<float>(&value)) { return "float(" + std::to_string(*x) + ")"; }
    if (auto x = std::any_cast<double>(&value)) { return "double(" + std::to_string(*x) + ")"; }
    if (auto x = std::any_cast<std::string>(&value)) { return "string(\"" + (*x) + "\")"; }
    if (auto x = std::any_cast<const char*>(&value)) { return *x; }
    return "other";
  }
  const uint32_t width() { return width_; }
  const uint32_t height() { return height_; }
  const uint64_t source_block_size() { return source_block_size_; }
  const uint64_t source_num_blocks() { return source_num_blocks_; }

  VideInputFragment(const std::string& source, const std::string& input_dir, const bool use_rdma)
      : source_(source), input_dir_(input_dir), use_rdma(use_rdma) {}

  void init() {
    if (source_ == "aja") {
      width_ = from_config("aja.width").as<uint32_t>();
      height_ = from_config("aja.height").as<uint32_t>();
      input_op_ = make_operator<ops::AJASourceOp>(
          "aja", from_config("aja"), from_config("external_source"));
      source_block_size_ = width_ * height_ * 4 * 4;
      source_num_blocks_ = use_rdma ? 3 : 4;
    } else if (source_ == "yuan") {
      width_ = from_config("yuan.width").as<uint32_t>();
      height_ = from_config("yuan.height").as<uint32_t>();
#ifdef YUAN_QCAP
      input_op_ = make_operator<ops::QCAPSourceOp>("yuan", from_config("yuan"));
#endif
      source_block_size_ = width_ * height_ * 4 * 4;
      source_num_blocks_ = use_rdma ? 3 : 4;
    } else {  // Replayer
      width_ = 854;
      height_ = 480;
      input_op_ = make_operator<ops::VideoStreamReplayerOp>(
          "replayer", from_config("replayer"), Arg("directory", input_dir_));
      source_block_size_ = width_ * height_ * 3 * 4;
      source_num_blocks_ = 2;
    }
  }

  void compose() override { add_operator(input_op_); }
};

class CloudInferenceFragment : public holoscan::Fragment {
 private:
  std::string config_key_name_;
  std::string model_dir_;
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  uint64_t source_block_size_ = 0;
  uint64_t source_num_blocks_ = 0;

 public:
  CloudInferenceFragment(const std::string& config_key_name, const std::string& model_dir,
                         const uint32_t width, const uint32_t height,
                         const uint64_t source_block_size, const uint64_t source_num_blocks)
      : config_key_name_(config_key_name),
        model_dir_(model_dir),
        width_(width),
        height_(height),
        source_block_size_(source_block_size),
        source_num_blocks_(source_num_blocks) {}

  void compose() override {
    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    auto format_converter = make_operator<ops::FormatConverterOp>(
        "format_converter",
        from_config(config_key_name_),
        Arg("pool") =
            make_resource<BlockMemoryPool>("pool", 1, source_block_size_, source_num_blocks_),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    const std::string model_file_path = model_dir_ + "/tool_loc_convlstm.onnx";
    const std::string engine_cache_dir = model_dir_ + "/engines";

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

    // const uint64_t tool_tracking_postprocessor_block_size = 107 * 60 * 7 * 4;
    // const uint64_t tool_tracking_postprocessor_num_blocks = 2;
    auto tool_tracking_postprocessor = make_operator<ops::ToolTrackingPostprocessorOp>(
        "tool_tracking_postprocessor",
        Arg("device_allocator") = make_resource<UnboundedAllocator>("device_allocator"),
        // make_resource<BlockMemoryPool>("device_allocator",
        //                                1,
        //                                tool_tracking_postprocessor_block_size,
        //                                tool_tracking_postprocessor_num_blocks),
        Arg("host_allocator") = make_resource<UnboundedAllocator>("host_allocator"));

    add_flow(format_converter, lstm_inferer);
    add_flow(lstm_inferer, tool_tracking_postprocessor, {{"tensor", "in"}});
  }
};

class VizFragment : public holoscan::Fragment {
 private:
  uint32_t width_ = 0;
  uint32_t height_ = 0;
  bool overlay_enabled_ = false;

 public:
  VizFragment(const uint32_t width, const uint32_t height, bool overlay_enabled)
      : width_(width), height_(height), overlay_enabled_(overlay_enabled) {}

  void compose() override {
    std::shared_ptr<BlockMemoryPool> visualizer_allocator;

    auto visualizer_operator =
        make_operator<ops::HolovizOp>("holoviz",
                                      from_config(overlay_enabled_ ? "holoviz_overlay" : "holoviz"),
                                      Arg("width") = width_,
                                      Arg("height") = height_,
                                      Arg("enable_render_buffer_input") = overlay_enabled_,
                                      Arg("enable_render_buffer_output") = overlay_enabled_,
                                      Arg("allocator") = visualizer_allocator);
    add_operator(visualizer_operator);
  }
};

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) { source_ = source; }

  void set_datapath(const std::string& path) { datapath_ = path; }

  void compose() override {
    using namespace holoscan;

    const bool overlay_enabled =
        (source_ != "replayer") && from_config("external_source.enable_overlay").as<bool>();
    const bool use_rdma = from_config("external_source.rdma").as<bool>();
    auto video_in = make_fragment<VideInputFragment>("video_in", source_, datapath_, use_rdma);
    auto video_in_fragment = std::dynamic_pointer_cast<VideInputFragment>(video_in);
    video_in_fragment->init();
    auto cloud_inference =
        make_fragment<CloudInferenceFragment>("inference",
                                              "format_converter_" + source_,
                                              datapath_,
                                              video_in_fragment->width(),
                                              video_in_fragment->height(),
                                              video_in_fragment->source_block_size(),
                                              video_in_fragment->source_num_blocks());
    auto viz = make_fragment<VizFragment>(
        "viz", video_in_fragment->width(), video_in_fragment->height(), overlay_enabled);

    // Flow definition
    if (source_ == "aja" || source_ == "yuan") {
      add_flow(video_in,
               cloud_inference,
               {{source_ + ".video_buffer_output", "format_converter.source_video"}});
    } else {
      add_flow(video_in, cloud_inference, {{"replayer", "format_converter"}});
    }
    add_flow(cloud_inference,
             viz,
             {{"tool_tracking_postprocessor.out_coords", "holoviz.receivers"},
              {"tool_tracking_postprocessor.out_mask", "holoviz.receivers"}});

    if (overlay_enabled) {
      // Overlay buffer flow between source and visualizer_operator
      add_flow(video_in, viz, {{"replayer.overlay_buffer_output", "holoviz.render_buffer_input"}});
      add_flow(viz, video_in, {{"holoviz.render_buffer_output", "replayer.overlay_buffer_input"}});
    } else {
      add_flow(video_in,
               viz,
               {{source_ != "replayer" ? "replayer.video_buffer_output" : "replayer.output",
                 "holoviz.receivers"}});
    }
  }

 private:
  std::string source_ = "replayer";
  std::string datapath_ = "data/endoscopy";
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
    HOLOSCAN_LOG_INFO("Using config file from {}", config_path.string());
  }

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);

  if (data_path != "") {
    app->set_datapath(data_path);
    HOLOSCAN_LOG_INFO("Using video from {}", data_path);
  } else {
    auto data_directory = std::getenv("HOLOSCAN_INPUT_PATH");
    if (data_directory != nullptr && data_directory[0] != '\0') {
      app->set_datapath(data_directory);
      HOLOSCAN_LOG_INFO("Using video from {}", data_directory);
    }
  }
  app->run();

  return 0;
}
