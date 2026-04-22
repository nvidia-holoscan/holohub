/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 Wayland Technologies. All rights reserved.
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

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>

#include "camera_calibration.hpp"
#include "mode_switcher_op.hpp"

#ifdef ATRACSYS_HAVE_LIVE_CAMERA
#include "master_source_op.hpp"
#include "point_cloud_filter_op.hpp"
#endif

namespace {

struct CommandLineOptions {
  std::string config_path;
  std::string data_path;
  std::string source{"replayer"};
};

void print_usage() {
  std::cout << "Usage: atracsys_visualizer [--config <yaml>] [--data <dir>] "
               "[--source replayer|live_camera]\n";
}

enum class ParseResult { kRun, kHelp, kError };

ParseResult parse_arguments(int argc, char** argv, CommandLineOptions& options) {
  static option long_options[] = {{"config", required_argument, nullptr, 'c'},
                                  {"data", required_argument, nullptr, 'd'},
                                  {"source", required_argument, nullptr, 's'},
                                  {"help", no_argument, nullptr, 'h'},
                                  {nullptr, 0, nullptr, 0}};

  while (true) {
    int option_index = 0;
    const int c = getopt_long(argc, argv, "c:d:s:h", long_options, &option_index);
    if (c == -1)
      break;

    switch (c) {
      case 'c':
        options.config_path = optarg;
        break;
      case 'd':
        options.data_path = optarg;
        break;
      case 's':
        options.source = optarg;
        break;
      case 'h':
        print_usage();
        return ParseResult::kHelp;
      default:
        print_usage();
        return ParseResult::kError;
    }
  }

  if (options.source != "replayer" && options.source != "live_camera") {
    throw std::runtime_error("Unsupported --source value: " + options.source);
  }

  return ParseResult::kRun;
}

auto find_config_node(holoscan::Application& app, const std::string& key) {
  for (const auto& yaml_node : app.config().yaml_nodes()) {
    const auto config_node = yaml_node[key];
    if (config_node && config_node.IsDefined()) { return config_node; }
  }
  throw std::runtime_error("Missing config entry: " + key);
}

std::shared_ptr<holoscan::ops::CameraCalibration> make_camera_calibration(
    holoscan::Application& app) {
  auto calibration = std::make_shared<holoscan::ops::CameraCalibration>();
  calibration->fx = static_cast<float>(app.from_config("camera_calibration_fx").as<double>());
  calibration->fy = static_cast<float>(app.from_config("camera_calibration_fy").as<double>());
  calibration->cx = static_cast<float>(app.from_config("camera_calibration_cx").as<double>());
  calibration->cy = static_cast<float>(app.from_config("camera_calibration_cy").as<double>());
  calibration->skew =
      static_cast<float>(app.from_config("camera_calibration_skew").as<double>());
  calibration->image_width = app.from_config("camera_calibration_image_width").as<int>();
  calibration->image_height = app.from_config("camera_calibration_image_height").as<int>();

  const auto distortion_node = find_config_node(app, "camera_calibration_distortion");
  if (!distortion_node.IsSequence() || distortion_node.size() != calibration->distortion.size()) {
    throw std::runtime_error("camera_calibration_distortion must contain exactly 5 coefficients");
  }
  for (size_t i = 0; i < distortion_node.size(); ++i) {
    calibration->distortion[i] = static_cast<float>(distortion_node[i].as<double>());
  }

  if (!calibration->valid()) {
    throw std::runtime_error("Atracsys visualizer calibration is invalid");
  }
  return calibration;
}

}  // namespace

class AtracsysVisualizerApp : public holoscan::Application {
 public:
  void set_source(std::string source) { source_ = std::move(source); }
  void set_data_path(std::string data_path) { data_path_ = std::move(data_path); }

  void compose() override {
    using namespace holoscan;

    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: creating image allocator");
    auto image_allocator =
        make_resource<BlockMemoryPool>("image_allocator", from_config("image_allocator"));
    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: creating visible RGB allocator");
    auto visible_rgb_allocator = make_resource<BlockMemoryPool>(
        "visible_rgb_allocator", from_config("visible_rgb_allocator"));
    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: creating display allocator");
    auto display_allocator =
        make_resource<BlockMemoryPool>("display_allocator", from_config("display_allocator"));
    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: creating structured points allocator");
    auto structured_points_allocator = make_resource<RMMAllocator>(
        "structured_points_allocator", from_config("structured_points_allocator"));
    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: creating replayer allocator");
    auto replayer_allocator =
        make_resource<RMMAllocator>("replayer_allocator", from_config("replayer_allocator"));
    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: creating CUDA stream pool");
    auto cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", from_config("cuda_stream_pool"));

    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: loading geometry path");
    const std::string geometry_path = from_config("geometry_path").as<std::string>();
    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: loading camera calibration");
    auto camera_calibration = make_camera_calibration(*this);

    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: creating holoviz and bayer demosaic operators");
    auto holoviz = make_operator<ops::HolovizOp>(
        "holoviz", from_config("holoviz"), Arg("cuda_stream_pool") = cuda_stream_pool);
    auto visible_bayer_demosaic =
        make_operator<ops::BayerDemosaicOp>("visible_bayer_demosaic",
                                            from_config("visible_bayer_demosaic"),
                                            Arg("pool") = visible_rgb_allocator,
                                            Arg("cuda_stream_pool") = cuda_stream_pool);

    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: creating mode switcher operator");
    auto mode_switcher = make_operator<holoscan::ops::AtracsysModeSwitcherOp>(
        "mode_switcher",
        from_config("mode_switcher"),
        Arg("display_allocator") = display_allocator,
        Arg("geometry_path") = geometry_path);
    mode_switcher->setCameraCalibration(std::move(camera_calibration));

    if (source_ == "live_camera") {
#ifdef ATRACSYS_HAVE_LIVE_CAMERA
      HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: creating live camera operators");
      auto async_condition = make_condition<AsynchronousCondition>("real_camera_async");
      auto camera_master = make_operator<holoscan::ops::AtracsysMasterSourceOp>(
          "atracsys_master",
          async_condition,
          from_config("real_camera"),
          Arg("image_allocator") = image_allocator,
          Arg("cuda_stream_pool") = cuda_stream_pool,
          Arg("structured_allocator") = structured_points_allocator,
          Arg("geometry_path") = geometry_path);

      auto point_cloud_filter = make_operator<holoscan::ops::PointCloudFilterOp>(
          "point_cloud_filter",
          from_config("point_cloud_filter"),
          Arg("structured_allocator") = structured_points_allocator,
          Arg("cuda_stream_pool") = cuda_stream_pool);

      add_flow(camera_master, visible_bayer_demosaic, {{"out_visible_base", "receiver"}});
      add_flow(visible_bayer_demosaic, mode_switcher, {{"transmitter", "in_visible_base"}});
      add_flow(camera_master, mode_switcher, {{"out_ir_base", "in_ir_base"}});
      add_flow(camera_master, mode_switcher, {{"out_marker_poses", "in_marker_poses"}});
      add_flow(camera_master,
               point_cloud_filter,
               {{"out_disparity", "in_disparity"}, {"out_q_matrix", "in_q_matrix"}});
      add_flow(
          point_cloud_filter, mode_switcher, {{"out_structured_points", "in_structured_points"}});
      add_flow(mode_switcher, camera_master, {{"out_hw_cmd", "in_hw_cmd"}});
#else
      throw std::runtime_error(
          "atracsys_visualizer was not built with the optional "
          "atracsys_camera operator");
#endif
    } else {
      HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: creating replay operators");
      auto replayer_visible =
          make_operator<ops::VideoStreamReplayerOp>("replayer_visible_base",
                                                    from_config("replayer_visible_base"),
                                                    Arg("allocator") = replayer_allocator,
                                                    Arg("directory") = data_path_);
      auto replayer_ir =
          make_operator<ops::VideoStreamReplayerOp>("replayer_ir_base",
                                                    from_config("replayer_ir_base"),
                                                    Arg("allocator") = replayer_allocator,
                                                    Arg("directory") = data_path_);
      auto replayer_structured_points =
          make_operator<ops::VideoStreamReplayerOp>("replayer_structured_points",
                                                    from_config("replayer_structured_points"),
                                                    Arg("allocator") = replayer_allocator,
                                                    Arg("directory") = data_path_);
      auto replayer_marker_poses =
          make_operator<ops::VideoStreamReplayerOp>("replayer_marker_poses",
                                                    from_config("replayer_marker_poses"),
                                                    Arg("allocator") = replayer_allocator,
                                                    Arg("directory") = data_path_);

      add_flow(replayer_visible, visible_bayer_demosaic, {{"output", "receiver"}});
      add_flow(visible_bayer_demosaic, mode_switcher, {{"transmitter", "in_visible_base"}});
      add_flow(replayer_ir, mode_switcher, {{"output", "in_ir_base"}});
      add_flow(replayer_structured_points, mode_switcher, {{"output", "in_structured_points"}});
      add_flow(replayer_marker_poses, mode_switcher, {{"output", "in_marker_poses"}});
    }

    HOLOSCAN_LOG_INFO("AtracsysVisualizerApp: connecting visualization flows");
    add_flow(mode_switcher, holoviz, {{"out_base", "receivers"}});
    add_flow(mode_switcher, holoviz, {{"out_overlay", "receivers"}});
    add_flow(mode_switcher, holoviz, {{"out_marker_points", "receivers"}});
    add_flow(mode_switcher, holoviz, {{"out_points", "receivers"}});
    add_flow(mode_switcher, holoviz, {{"out_mode_text", "receivers"}});
    add_flow(mode_switcher, holoviz, {{"out_fiducial_text_coords", "receivers"}});
    add_flow(mode_switcher, holoviz, {{"out_specs", "input_specs"}});
  }

 private:
  std::string source_{"replayer"};
  std::string data_path_{"data/atracsys_visualizer"};
};

int main(int argc, char** argv) {
  try {
    CommandLineOptions options;
    const auto parse_result = parse_arguments(argc, argv, options);
    if (parse_result == ParseResult::kHelp) {
      return 0;
    }
    if (parse_result == ParseResult::kError) {
      return 1;
    }

    if (options.config_path.empty()) {
      options.config_path = "atracsys_visualizer.yaml";
    }

    if (options.data_path.empty()) {
      if (const char* env_path = std::getenv("HOLOSCAN_INPUT_PATH")) {
        options.data_path =
            std::string(env_path) + "/atracsys_visualizer";
      } else {
        options.data_path = "data/atracsys_visualizer";
      }
    }

    auto app = holoscan::make_application<AtracsysVisualizerApp>();
    app->set_source(options.source);
    app->set_data_path(options.data_path);
    app->config(options.config_path);
    app->run();

    return 0;
  } catch (const std::exception& e) {
    HOLOSCAN_LOG_ERROR("Application failed with exception: {}", e.what());
    return 1;
  } catch (...) {
    HOLOSCAN_LOG_ERROR("Application failed with unknown exception.");
    return 1;
  }
}
