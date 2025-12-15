/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights
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
#include <iostream>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/v4l2_video_capture/v4l2_video_capture.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include "crop.h"
#include "heat_map.h"
#include "split_video.h"
#include "undistort_rectify.h"
#include "vpi_stereo.h"

class VPIStereoApp;

class VPIStereoApp : public holoscan::Application {
 private:
  std::string stereo_calibration_;
  std::string source_;
  std::string datapath_;

 public:
  void set_datapath(const std::string& path) { datapath_ = path; }

  VPIStereoApp(std::string source, std::string file) : source_(source), stereo_calibration_(file) {}
  void compose() override {
    using namespace holoscan;

    // video stream, format conversion, split single stereo video frame into left and right
    // frames.
    std::shared_ptr<Operator> video_source;
    std::string source_output;
    auto in_dtype = Arg("in_dtype", std::string("rgba8888"));

    if (source_ == "v4l2") {
      video_source = make_operator<ops::V4L2VideoCaptureOp>(
          "source",
          from_config("v4l2"),
          Arg("allocator") = make_resource<UnboundedAllocator>("pool_v4l2"));
      source_output = "signal";
    } else if (source_ == "replayer") {
      video_source = make_operator<ops::VideoStreamReplayerOp>(
          "replayer",
          from_config("replayer"),
          Arg("directory", datapath_),
          Arg("allocator") = make_resource<UnboundedAllocator>("pool_replayer"));
      source_output = "output";
      in_dtype = Arg("in_dtype", std::string("rgb888"));
    } else {
      throw std::runtime_error("Unsupported video source");
    }

    auto v4l2_converter_pool =
        Arg("pool", make_resource<holoscan::UnboundedAllocator>("pool_v4l2"));
    /// TODO: VPI would prefer single channel 8b image or NV12, but neither the V4L2 capture or
    /// format converter can support that today.  VPI has RGBA8888 -> Mono format conversion support
    /// via CUDA or VIC, but RGB888 support only via CUDA.
    auto out_dtype = Arg("out_dtype", std::string("rgb888"));
    auto v4l2_converter = make_operator<ops::FormatConverterOp>(
        "converter", in_dtype, out_dtype, v4l2_converter_pool);
    auto splitter = make_operator<ops::SplitVideoOp>(
        "splitter", Arg("stereo_video_layout", STEREO_VIDEO_HORIZONTAL));

    // load camera calibration data
    YAML::Node calibration = YAML::LoadFile(stereo_calibration_);
    std::vector<float> M1 = calibration["M1"].as<std::vector<float>>();
    std::vector<float> d1 = calibration["d1"].as<std::vector<float>>();
    std::vector<float> M2 = calibration["M2"].as<std::vector<float>>();
    std::vector<float> d2 = calibration["d2"].as<std::vector<float>>();
    std::vector<float> R = calibration['R'].as<std::vector<float>>();
    std::vector<float> t = calibration['t'].as<std::vector<float>>();
    int width = calibration["width"].as<int>();
    int height = calibration["height"].as<int>();

    // solve transforms for rectification using calibration.
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

    // generate flow maps from transforms
    auto rectification_map1 = std::make_shared<ops::UndistortRectifyOp::RectificationMap>(
        &M1[0], &d1[0], R1_float, P1_float, width, height);
    auto rectification_map2 = std::make_shared<ops::UndistortRectifyOp::RectificationMap>(
        &M2[0], &d2[0], R2_float, P2_float, width, height);

    // init rectification operators
    auto rectifier1 = make_operator<ops::UndistortRectifyOp>("rectifier1");
    rectifier1->setRectificationMap(rectification_map1);
    auto rectifier2 = make_operator<ops::UndistortRectifyOp>("rectifier2");
    rectifier2->setRectificationMap(rectification_map2);

    // adjust frames for display
    auto crop_color = make_operator<ops::CropOp>("crop_color",
                                                 Arg("x", roi[0]),
                                                 Arg("y", roi[1]),
                                                 Arg("width", roi[2]),
                                                 Arg("height", roi[3]));

    auto crop_disparity = make_operator<ops::CropOp>("crop_disparity",
                                                     Arg("x", roi[0]),
                                                     Arg("y", roi[1]),
                                                     Arg("width", roi[2]),
                                                     Arg("height", roi[3]));
    auto heatmap = make_operator<ops::HeatmapOp>("heatmap", from_config("heatmap"));

    // window for viz of left camera + left disparity
    auto merger = make_operator<ops::MergeVideoOp>(
        "merger", Arg("stereo_video_layout", STEREO_VIDEO_HORIZONTAL));
    auto holoviz = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));

    // Rectification
    add_flow(video_source, v4l2_converter, {{source_output, "source_video"}});
    add_flow(v4l2_converter, splitter, {{"tensor", "input"}});
    add_flow(splitter, rectifier1, {{"output1", "input"}});
    add_flow(splitter, rectifier2, {{"output2", "input"}});

    // VPI processing
    auto vpi_stereo = make_operator<ops::VPIStereoOp>("vpi_stereo", from_config("vpi_stereo"));

    // Compute stereo disparity
    add_flow(rectifier1, vpi_stereo, {{"output", "input1"}});
    add_flow(rectifier2, vpi_stereo, {{"output", "input2"}});

    // Adjust stereo output for display
    add_flow(vpi_stereo, crop_disparity, {{"output", "input"}});

    // Visualization
    add_flow(rectifier1, crop_color, {{"output", "input"}});
    add_flow(crop_disparity, heatmap, {{"output", "input"}});
    add_flow(crop_color, merger, {{"output", "input1"}});
    add_flow(heatmap, merger, {{"output", "input2"}});
    add_flow(merger, holoviz, {{"output", "receivers"}});
  }
};

void print_usage() {
  std::cout << "Usage: program [--config <config-file>]\n"
               "               [--source <v4l2|replayer>]\n"
               "               [--data <data-directory>]\n"
               "               [--stereo-calibration <stereo-calibration-file>]\n";
}

void parse_arguments(int argc, char* argv[], std::string& data_path, std::string& config_file,
                     std::string& source, std::string& stereo_file) {
  int option_index = 0;
  static struct option long_options[] = {{"config", required_argument, 0, 'c'},
                                         {"data", required_argument, 0, 'd'},
                                         {"source", required_argument, 0, 's'},
                                         {"stereo-calibration", required_argument, 0, 't'},
                                         {0, 0, 0, 0}};

  int c;
  while ((c = getopt_long(argc, argv, "c:d:s:t:", long_options, &option_index)) != -1) {
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
      data_directory = std::string(input_path) + "/vpi_stereo";
    } else if (std::filesystem::is_directory(std::filesystem::current_path() / "data/vpi_stereo")) {
      data_directory = std::string((std::filesystem::current_path() / "data/vpi_stereo").c_str());
    } else {
      HOLOSCAN_LOG_ERROR(
          "Input data not provided. Use --data or set HOLOSCAN_INPUT_PATH environment variable.");
      exit(1);
    }
  }

  if (stereo_cal.empty()) {
    stereo_cal = data_directory + "/stereo_calibration.yaml";
  }

  if (config_file.empty()) {
    auto default_path = std::filesystem::canonical(argv[0]).parent_path();
    default_path /= std::filesystem::path("vpi_stereo.yaml");
    config_file = default_path.string();
  }

  if (source.empty()) {
    source = "replayer";
  }

  auto app = holoscan::make_application<VPIStereoApp>(source, stereo_cal);
  app->config(config_file);
  app->set_datapath(data_directory);
  app->run();
  return 0;
}
