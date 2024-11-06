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
#include <holoscan/operators/aja_source/aja_source.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/inference_processor/inference_processor.hpp>
#include <holoscan/operators/video_stream_recorder/video_stream_recorder.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include <holoscan/version_config.hpp>

#include <visualizer_icardio.hpp>

#define HOLOSCAN_VERSION \
  (HOLOSCAN_VERSION_MAJOR * 10000 + HOLOSCAN_VERSION_MINOR * 100 + HOLOSCAN_VERSION_PATCH)

class App : public holoscan::Application {
 public:
  void set_source(const std::string& source) {
    if (source == "aja") { is_aja_source_ = true; }
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

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream");

    if (is_aja_source_) {
      source = make_operator<ops::AJASourceOp>("aja", from_config("aja"));
    } else {
      source = make_operator<ops::VideoStreamReplayerOp>(
          "replayer", from_config("replayer"), Arg("directory", datapath));
#if HOLOSCAN_VERSION >= 20600
      // the RMMAllocator supported since v2.6 is much faster than the default UnboundAllocator
      source->add_arg(Arg("allocator", make_resource<RMMAllocator>("video_replayer_allocator")));
#endif
    }

    auto in_dtype = is_aja_source_ ? std::string("rgba8888") : std::string("rgb888");
    auto in_components = is_aja_source_ ? 4 : 3;
    // FormatConverterOp needs an temporary buffer if converting from RGBA
    auto format_convert_pool_blocks = (in_components == 4) ? 4 : 3;
    auto plax_cham_pre =
        make_operator<ops::FormatConverterOp>("plax_cham_pre",
                                              from_config("plax_cham_pre"),
                                              Arg("in_dtype") = in_dtype,
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "plax_cham_pre_pool",
                                                  (int32_t)nvidia::gxf::MemoryStorageType::kDevice,
                                                  320 * 320 * sizeof(float) * in_components,
                                                  format_convert_pool_blocks),
                                              Arg("cuda_stream_pool") = cuda_stream_pool);

    auto aortic_ste_pre =
        make_operator<ops::FormatConverterOp>("aortic_ste_pre",
                                              from_config("aortic_ste_pre"),
                                              Arg("in_dtype") = in_dtype,
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "aortic_ste_pre_pool",
                                                  (int32_t)nvidia::gxf::MemoryStorageType::kDevice,
                                                  300 * 300 * sizeof(float) * in_components,
                                                  format_convert_pool_blocks),
                                              Arg("cuda_stream_pool") = cuda_stream_pool);

    auto b_mode_pers_pre =
        make_operator<ops::FormatConverterOp>("b_mode_pers_pre",
                                              from_config("b_mode_pers_pre"),
                                              Arg("in_dtype") = in_dtype,
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "b_mode_pers_pre_pool",
                                                  (int32_t)nvidia::gxf::MemoryStorageType::kDevice,
                                                  320 * 240 * sizeof(float) * in_components,
                                                  format_convert_pool_blocks),
                                              Arg("cuda_stream_pool") = cuda_stream_pool);

    ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("plax_chamber", datapath + "/plax_chamber.onnx");
    model_path_map.insert("aortic_stenosis", datapath + "/aortic_stenosis.onnx");
    model_path_map.insert("bmode_perspective", datapath + "/bmode_perspective.onnx");

    const auto plax_chamber_output_size = 320 * 320 * sizeof(float) * 6;
    const auto aortic_stenosis_output_size = sizeof(float) * 2;
    const auto bmode_perspective_output_size = sizeof(float) * 1;
    auto block_size =
        std::max(plax_chamber_output_size,
                 std::max(aortic_stenosis_output_size, bmode_perspective_output_size));
    auto multiai_inference =
        make_operator<ops::InferenceOp>("multiai_inference",
                                        from_config("multiai_inference"),
                                        Arg("model_path_map", model_path_map),
                                        Arg("allocator") = make_resource<BlockMemoryPool>(
                                            "multiai_inference_allocator",
                                            (int32_t)nvidia::gxf::MemoryStorageType::kDevice,
                                            block_size,
                                            2 * 3),
                                        Arg("cuda_stream_pool") = cuda_stream_pool);

    //  version 2.6 supports the CUDA version of `max_per_channel_scaled`
    const bool supports_cuda_processing =
#if HOLOSCAN_VERSION >= 20600
        true;
#else
        false;
#endif
    auto multiai_postprocessor = make_operator<ops::InferenceProcessorOp>(
        "multiai_postprocessor",
        from_config("multiai_postprocessor"),
        Arg("input_on_cuda", supports_cuda_processing),
        Arg("output_on_cuda", supports_cuda_processing),
        Arg("allocator") =
            make_resource<BlockMemoryPool>("multiai_postprocessor_allocator",
                                           (int32_t)nvidia::gxf::MemoryStorageType::kDevice,
                                           // 2 float coordinates, 6 categories
                                           2 * sizeof(float) * 6,
                                           1),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    auto visualizer_icardio = make_operator<ops::VisualizerICardioOp>(
        "visualizer_icardio",
        from_config("visualizer_icardio"),
        Arg("data_dir") = datapath,
        Arg("allocator") =
            make_resource<BlockMemoryPool>("visualizer_icardio_allocator",
                                           (int32_t)nvidia::gxf::MemoryStorageType::kDevice,
                                           // max from VisualizerICardioOp::tensor_to_shape_
                                           320 * 320 * 4 * sizeof(uint8_t),
                                           1 * 8),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz",
        from_config("holoviz"),
        Arg("enable_render_buffer_output") = (record_type_ == Record::VISUALIZER),
        Arg("allocator") =
            make_resource<BlockMemoryPool>("visualizer_allocator",
                                           (int32_t)nvidia::gxf::MemoryStorageType::kDevice,
                                           // max from VisualizerICardioOp::tensor_to_shape_
                                           320 * 320 * 4 * sizeof(uint8_t),
                                           1 * 8),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    // Add recording operators
    if (record_type_ != Record::NONE) {
      if (((record_type_ == Record::INPUT) && is_aja_source_) ||
          (record_type_ == Record::VISUALIZER)) {
        recorder_format_converter = make_operator<ops::FormatConverterOp>(
            "recorder_format_converter",
            from_config("recorder_format_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool",
                                               (int32_t)nvidia::gxf::MemoryStorageType::kDevice,
                                               320 * 320 * 4 * sizeof(uint8_t),
                                               1 * 8));
      }
      recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));
    }

    // Flow definition
    const std::string source_port_name = is_aja_source_ ? "video_buffer_output" : "";
    add_flow(source, plax_cham_pre, {{source_port_name, ""}});
    add_flow(source, aortic_ste_pre, {{source_port_name, ""}});
    add_flow(source, b_mode_pers_pre, {{source_port_name, ""}});
    add_flow(source, holoviz, {{source_port_name, "receivers"}});

    add_flow(plax_cham_pre, multiai_inference, {{"", "receivers"}});
    add_flow(aortic_ste_pre, multiai_inference, {{"", "receivers"}});
    add_flow(b_mode_pers_pre, multiai_inference, {{"", "receivers"}});

    add_flow(multiai_inference, multiai_postprocessor, {{"transmitter", "receivers"}});
    add_flow(multiai_postprocessor, visualizer_icardio, {{"transmitter", "receivers"}});

    add_flow(visualizer_icardio, holoviz, {{"keypoints", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"keyarea_1", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"keyarea_2", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"keyarea_3", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"keyarea_4", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"keyarea_5", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"lines", "receivers"}});
    add_flow(visualizer_icardio, holoviz, {{"logo", "receivers"}});

    if (record_type_ == Record::INPUT) {
      if (is_aja_source_) {
        add_flow(source, recorder_format_converter, {{source_port_name, "source_video"}});
        add_flow(recorder_format_converter, recorder);
      } else {
        add_flow(source, recorder);
      }
    } else if (record_type_ == Record::VISUALIZER) {
      add_flow(holoviz, recorder_format_converter, {{"render_buffer_output", "source_video"}});
      add_flow(recorder_format_converter, recorder);
    }
  }

 private:
  bool is_aja_source_ = false;
  Record record_type_ = Record::NONE;
  std::string datapath = "data/multiai_ultrasound";
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

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) { return 1; }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/multiai_ultrasound.yaml";
    app->config(config_path);
  }

  auto source = app->from_config("source").as<std::string>();
  app->set_source(source);

  auto record_type = app->from_config("record_type").as<std::string>();
  app->set_record(record_type);

  if (data_path != "") app->set_datapath(data_path);
  app->run();

  return 0;
}
