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
#include <npp.h>
#include <Eigen/Dense>
#include <iostream>

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
#include "tracking_postprocessor.h"
#include "undistort_rectify.h"

class StereoDepthApp;

class StereoDepthApp : public holoscan::Application {
 private:
  std::string stereo_calibration_;

 public:
  explicit StereoDepthApp(std::string file) : stereo_calibration_(file) {}
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

    auto source = make_operator<ops::V4L2VideoCaptureOp>(
        "source",
        from_config("source"),
        Arg("allocator") = make_resource<UnboundedAllocator>("pool_v4l2"));

    auto v4l2_converter_pool =
        Arg("pool", make_resource<holoscan::UnboundedAllocator>("pool_v4l2"));
    auto in_dtype = Arg("in_dtype", std::string("rgba8888"));
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
    auto merger = make_operator<ops::MergeVideoOp>(
        "merger", Arg("stereo_video_layout", STEREO_VIDEO_HORIZONTAL));

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

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);
    uint64_t source_block_size = width * height * 4 * 4;
    uint64_t source_num_blocks = 4;
    auto yolo_format_converter =
        make_operator<ops::FormatConverterOp>("yolo_format_converter",
                                              from_config("yolo_format_converter"),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks),
                                              Arg("cuda_stream_pool") = cuda_stream_pool);

    auto yolo_inference = make_operator<ops::InferenceOp>(
        "yolo_inference",
        from_config("yolo_inference"),
        Arg("allocator") = make_resource<UnboundedAllocator>("pool_yolo"));

    auto tracking_postprocessor =
        make_operator<ops::TrackingPostprocessor>("tracking_postprocessor");
    auto tracking_postprocessor_heatmap =
        make_operator<ops::TrackingPostprocessor>("tracking_postprocessor_heatmap");

    ////////////////////////////
    // Connect the operators   //
    ////////////////////////////

    // Rectification
    add_flow(source, v4l2_converter, {{"signal", "source_video"}});
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

    // YOLO detection
    add_flow(rectifier1, crop_color, {{"output", "input"}});
    add_flow(crop_color, yolo_format_converter, {{"output", "source_video"}});
    add_flow(yolo_format_converter, yolo_inference, {{"tensor", "receivers"}});
    add_flow(yolo_inference, tracking_postprocessor, {{"transmitter", "yolo_input"}});
    add_flow(crop_color, tracking_postprocessor, {{"output", "image"}});

    // Visualization
    // draw yolo boxes on disparity heatmap as well
    add_flow(yolo_inference, tracking_postprocessor_heatmap, {{"transmitter", "yolo_input"}});
    add_flow(heatmap_ess, tracking_postprocessor_heatmap, {{"output", "image"}});

    add_flow(tracking_postprocessor_heatmap, merger, {{"output", "input1"}});
    add_flow(tracking_postprocessor, merger, {{"output", "input2"}});
    add_flow(merger, holoviz, {{"output", "receivers"}});
    add_flow(heatmap_ess, holoviz, {{"output", "receivers"}});
  }
};

void print_usage() {
  std::cout << "Usage: program [--config <config-file>] [--stereo <stereo-calibration-file>]\n";
}

void parse_arguments(int argc, char* argv[], char*& config_file, char*& stereo_file) {
  int option_index = 0;
  static struct option long_options[] = {{"config", required_argument, 0, 'c'},
                                         {"stereo-calibration", required_argument, 0, 's'},
                                         {0, 0, 0, 0}};

  int c;
  while ((c = getopt_long(argc, argv, "c:s:", long_options, &option_index)) != -1) {
    switch (c) {
      case 'c':
        config_file = optarg;
        break;
      case 's':
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
  char* config_file = nullptr;
  char* stereo_cal = nullptr;
  std::string stereo_cal_string;
  std::string config_file_string;
  parse_arguments(argc, argv, config_file, stereo_cal);
  if (stereo_cal) {
    stereo_cal_string = stereo_cal;
  } else {
    auto default_path = std::filesystem::canonical(argv[0]).parent_path();
    default_path /= std::filesystem::path("stereo_calibration.yaml");
    stereo_cal_string = default_path.string();
  }
  if (config_file) {
    config_file_string = config_file;
  } else {
    auto default_path = std::filesystem::canonical(argv[0]).parent_path();
    default_path /= std::filesystem::path("stereo_vision.yaml");
    config_file_string = default_path.string();
  }
  auto app = holoscan::make_application<StereoDepthApp>(stereo_cal_string);
  app->config(config_file_string);
  app->run();
  return 0;
}
