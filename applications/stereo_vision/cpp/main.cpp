/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <npp.h>
#include <Eigen/Dense>
#include <iostream>
#include <dlfcn.h>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/inference_processor/inference_processor.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/utils/cuda_stream_handler.hpp>

#include <holoscan/operators/format_converter/format_converter.hpp>
#include "crop.h"
#include "ess_processor.h"
#include "heat_map.h"
#include "split_video.h"
#include "stereo_depth_kernels.h"
#include "undistort_rectify.h"

class StereoDepthApp;

class StereoDepthApp : public holoscan::Application {
 private:
  std::string source_;
  std::string stereo_calibration_;
  std::string datapath_;

 public:
  explicit StereoDepthApp(std::string source, std::string file) :
      source_(source), stereo_calibration_(file) {}

  void set_datapath(const std::string& path) { datapath_ = path; }

  void compose() override {
    using namespace holoscan;
    YAML::Node calibration = YAML::LoadFile(stereo_calibration_);
    std::vector<float> M1 = calibration["M1"].as<std::vector<float>>();
    std::vector<float> d1 = calibration["d1"].as<std::vector<float>>();
    std::vector<float> M2 = calibration["M2"].as<std::vector<float>>();
    std::vector<float> d2 = calibration["d2"].as<std::vector<float>>();
    std::vector<float> R = calibration['R'].as<std::vector<float>>();
    std::vector<float> t = calibration["t"].as<std::vector<float>>();
    int width = calibration["width"].as<int>();
    int height = calibration["height"].as<int>();

    std::shared_ptr<Operator> source;
    std::string source_output;

    auto in_dtype = Arg("in_dtype", std::string("rgba8888"));

    if (source_ == "v4l2") {
      source = make_operator<ops::V4L2VideoCaptureOp>(
        "v4l2",
        from_config("v4l2"),
        Arg("allocator") = make_resource<UnboundedAllocator>("pool_source"));
      source_output = "signal";
    } else if (source_ == "replayer") {
      source = make_operator<ops::VideoStreamReplayerOp>(
        "replayer",
        from_config("replayer"),
        Arg("directory", datapath_),
        Arg("allocator") = make_resource<UnboundedAllocator>("pool_source"));
      source_output = "output";
      in_dtype = Arg("in_dtype", std::string("rgb888"));
    } else {
      std::cout << "Unsupported input source" << std::endl;
      exit(1);
    }

    auto v4l2_converter_pool =
        Arg("pool", make_resource<holoscan::UnboundedAllocator>("pool_converter"));
    auto out_dtype = Arg("out_dtype", std::string("rgb888"));
    auto v4l2_converter = make_operator<ops::FormatConverterOp>(
        "converter", in_dtype, out_dtype, v4l2_converter_pool);

    auto splitter = make_operator<ops::SplitVideoOp>(
        "splitter", Arg("stereo_video_layout", STEREO_VIDEO_HORIZONTAL));

    float R1_float[9];
    float R2_float[9];
    float P1_float[12];
    float P2_float[12];
    float Q_float[16];
    int roi[4];

    ops::UndistortRectifyOp::stereoRectify(&M1[0],
                                           &d1[0],
                                           &M2[0],
                                           &d2[0],
                                           &R[0],
                                           &t[0],
                                           width,
                                           height,
                                           R1_float,
                                           R2_float,
                                           P1_float,
                                           P2_float,
                                           Q_float,
                                           roi);

    auto rectification_map1 = std::make_shared<ops::UndistortRectifyOp::RectificationMap>(
        &M1[0], &d1[0], R1_float, P1_float, width, height);
    auto rectification_map2 = std::make_shared<ops::UndistortRectifyOp::RectificationMap>(
        &M2[0], &d2[0], R2_float, P2_float, width, height);

    auto rectifier1 = make_operator<ops::UndistortRectifyOp>("rectifier1");
    rectifier1->setRectificationMap(rectification_map1);
    auto rectifier2 = make_operator<ops::UndistortRectifyOp>("rectifier2");
    rectifier2->setRectificationMap(rectification_map2);

    auto holoviz = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));

    auto ess_preprocessor =
        make_operator<ops::ESSPreprocessorOp>("ess_preprocessor", from_config("ess_preprocessor"));
    auto ess_postprocessor = make_operator<ops::ESSPostprocessorOp>(
        "ess_postprocessor", Arg("width", width), Arg("height", height));

    auto crop_color = make_operator<ops::CropOp>("crop_color",
                                                 Arg("x", roi[0]),
                                                 Arg("y", roi[1]),
                                                 Arg("width", roi[2]),
                                                 Arg("height", roi[3]));

    auto crop_disparity_ess = make_operator<ops::CropOp>("crop_disparity_ess",
                                                         Arg("x", roi[0]),
                                                         Arg("y", roi[1]),
                                                         Arg("width", roi[2]),
                                                         Arg("height", roi[3]));

    auto heatmap_ess = make_operator<ops::HeatmapOp>("heatmap_ess", from_config("heatmap_ess"));

    auto ess_inference = make_operator<ops::InferenceOp>(
        "inference",
        from_config("ess_inference"),
        Arg("allocator") = make_resource<UnboundedAllocator>("pool_ess"));

    ////////////////////////////
    // Connect the operators   //
    ////////////////////////////

    // Rectification
    add_flow(source, v4l2_converter, {{source_output, "source_video"}});
    add_flow(v4l2_converter, splitter, {{"tensor", "input"}});
    add_flow(splitter, rectifier1, {{"output1", "input"}});
    add_flow(splitter, rectifier2, {{"output2", "input"}});

    // // Stereo Disparity Estimation
    add_flow(rectifier1, ess_preprocessor, {{"output", "input1"}});
    add_flow(rectifier2, ess_preprocessor, {{"output", "input2"}});
    add_flow(ess_preprocessor, ess_inference, {{"output", "receivers"}});
    add_flow(ess_inference, ess_postprocessor, {{"transmitter", "input"}});
    add_flow(ess_postprocessor, crop_disparity_ess, {{"output", "input"}});
    add_flow(crop_disparity_ess, heatmap_ess, {{"output", "input"}});
    add_flow(heatmap_ess, holoviz, {{"output", "receivers"}});
  }
};

void print_usage() {
  std::cout << "Usage: program [--config <config-file>] [--stereo <stereo-calibration-file>]\n";
}

void parse_arguments(int argc, char* argv[], std::string& data_path, std::string& config_file,
        std::string& source, std::string& stereo_file) {
  int option_index = 0;
  static struct option long_options[] = {{"config", required_argument, 0, 'c'},
                                         {"source", required_argument, 0, 's'},
                                         {"data", required_argument, 0, 'd'},
                                         {"stereo-calibration", required_argument, 0, 't'},
                                         {0, 0, 0, 0}};

  int c;
  while ((c = getopt_long(argc, argv, "c:s:t:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'c':
        config_file = optarg;
        break;
      case 'd':
        data_path = optarg;
        break;
      case 's':
        if (strcmp(optarg, "replayer") != 0 && strcmp(optarg, "v4l2") != 0) {
          std::cerr << "Error: Invalid value for --source. Allowed values: {replayer, v4l2}.\n";
          print_usage();
          exit(1);
        }
        source = optarg;
        break;
      case 't':
        stereo_file = optarg;
        break;
      case '?':
        print_usage();
        exit(1);
      default:
        print_usage();
        exit(1);
    }
  }
}

int main(int argc, char** argv) {
  std::string data_directory, config_file, source, stereo_cal;

  parse_arguments(argc, argv, data_directory, config_file, source, stereo_cal);

  if (data_directory.empty()) {
    auto input_path = std::getenv("HOLOSCAN_INPUT_PATH");
    if (input_path != nullptr && input_path[0] != '\0') {
      data_directory = std::string(input_path) + "/stereo_vision";
    } else if (std::filesystem::is_directory(
          std::filesystem::current_path() / "data/stereo_vision")) {
      data_directory = std::string(
          (std::filesystem::current_path() / "data/stereo_vision").c_str());
    } else {
      HOLOSCAN_LOG_ERROR(
          "Input data not provided. Use --data or set HOLOSCAN_INPUT_PATH environment variable.");
      exit(1);
    }
  }

  // Load TensorRT plugins with RTLD_LOCAL to avoid GXF extension initialization
  std::string plugin_path = data_directory + "/ess_plugins.so";
  void* plugin_handle = dlopen(plugin_path.c_str(), RTLD_LAZY | RTLD_LOCAL);
  if (!plugin_handle) {
    HOLOSCAN_LOG_ERROR("Failed to load TensorRT plugins from {}: {}", plugin_path, dlerror());
  }

  if (stereo_cal.empty()) {
    stereo_cal = data_directory + "/stereo_calibration.yaml";
  }

  if (config_file.empty()) {
    auto default_path = std::filesystem::canonical(argv[0]).parent_path();
    default_path /= std::filesystem::path("stereo_vision.yaml");
    config_file = default_path.string();
  }

  if (source.empty()) {
    source = "replayer";
  }

  auto app = holoscan::make_application<StereoDepthApp>(source, stereo_cal);
  app->config(config_file);
  app->set_datapath(data_directory);
  app->run();

  if (plugin_handle) {
    dlclose(plugin_handle);
  }

  return 0;
}
