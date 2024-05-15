/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "convert_depth_to_screen_space_op.hpp"
#include "holoscan/holoscan.hpp"
#include "xr_begin_frame_op.hpp"
#include "xr_end_frame_op.hpp"
#include "xr_transform_control_op.hpp"
#include "xr_transform_render_op.hpp"

#include "volume_loader.hpp"
#include "volume_renderer.hpp"

#include <getopt.h>
#include <nlohmann/json.hpp>
#include <string>

/**
 * Dummy YAML convert function for shared data type
 */
template <>
struct YAML::convert<std::shared_ptr<nlohmann::json>> {
  static Node encode(const std::shared_ptr<nlohmann::json>& data) {
    holoscan::log_error("YAML conversion not supported");
    return Node();
  }

  static bool decode(const Node& node, std::shared_ptr<nlohmann::json>& data) {
    holoscan::log_error("YAML conversion not supported");
    return false;
  }
};

template <>
struct YAML::convert<std::shared_ptr<std::array<nvidia::gxf::Vector2f, 3>>> {
  static Node encode(const std::shared_ptr<std::array<nvidia::gxf::Vector2f, 3>>& data) {
    holoscan::log_error("YAML conversion not supported");
    return Node();
  }

  static bool decode(const Node& node,
                     std::shared_ptr<std::array<nvidia::gxf::Vector2f, 3>>& data) {
    holoscan::log_error("YAML conversion not supported");
    return false;
  }
};

namespace holoscan::ops {

/**
 * Source Op for typed shared data.
 */
template <typename T>
class SharedDataSourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SharedDataSourceOp);

  void initialize() override {
    register_converter<std::shared_ptr<T>>();
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override {
    spec.param(shared_, "shared", "shared", "Shared Type");
    spec.output<T>("out");
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    output.emit(*shared_.get(), "out");
  }

 private:
  Parameter<std::shared_ptr<T>> shared_;
};

/**
 * Sink Op for typed shared data.
 */
template <typename T>
class SharedDataSinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SharedDataSinkOp);

  void setup(OperatorSpec& spec) override {
    spec.param(shared_, "shared", "shared", "Shared Type");
    spec.input<T>("in");
  }

  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override {
    auto message = input.receive<T>("in");
    *shared_.get() = message.value();
  }

 private:
  Parameter<std::shared_ptr<T>> shared_;
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  App(const std::string& render_config_file, const std::string& write_config_file,
      const std::string& density_volume_file, const std::string& mask_volume_file,
      bool enable_eye_tracking)
      : render_config_file_(render_config_file),
        write_config_file_(write_config_file),
        density_volume_file_(density_volume_file),
        mask_volume_file_(mask_volume_file),
        enable_eye_tracking_(enable_eye_tracking) {}
  App() = delete;

  void compose() override {
    std::shared_ptr<holoscan::openxr::XrSession> xr_session =
        make_resource<holoscan::openxr::XrSession>(
            "xr_session",
            holoscan::Arg("application_name") = std::string("Render Volume XR"),
            holoscan::Arg("application_version") = 1u,
            holoscan::Arg("near_z", 0.33f),
            holoscan::Arg("far_z", 10.f));
    // resources are lazy initialized by Holoscan but we need the session initialized here to get
    // the display size
    xr_session->initialize();

    auto xr_begin_frame = make_operator<holoscan::openxr::XrBeginFrameOp>(
        "xr_begin_frame",
        holoscan::Arg("session") = xr_session,
        holoscan::Arg("enable_eye_tracking") = enable_eye_tracking_);
    auto xr_end_frame = make_operator<holoscan::openxr::XrEndFrameOp>(
        "xr_end_frame", holoscan::Arg("session") = xr_session);

    auto xr_transform_controller =
        make_operator<holoscan::openxr::XrTransformControlOp>("xr_transform_controller");
    auto xr_transform_renderer = make_operator<holoscan::openxr::XrTransformRenderOp>(
        "xr_transform_render",
        holoscan::Arg("display_width", xr_session->display_width()),
        holoscan::Arg("display_height", xr_session->display_height()),
        holoscan::Arg("config_file", render_config_file_));

    auto density_volume_loader = make_operator<holoscan::ops::VolumeLoaderOp>(
        "density_volume_loader",
        holoscan::Arg("file_name", density_volume_file_),
        holoscan::Arg("allocator", make_resource<holoscan::UnboundedAllocator>("allocator")),
        // the loader will executed only once to load the volume
        make_condition<holoscan::CountCondition>("count-condition", 1));

    std::shared_ptr<holoscan::ops::VolumeLoaderOp> mask_volume_loader;
    if (!mask_volume_file_.empty()) {
      mask_volume_loader = make_operator<holoscan::ops::VolumeLoaderOp>(
          "mask_volume_loader",
          holoscan::Arg("file_name", mask_volume_file_),
          holoscan::Arg("allocator", make_resource<holoscan::UnboundedAllocator>("allocator")),
          // the loader will executed only once to load the volume
          make_condition<holoscan::CountCondition>("count-condition", 1));
    }

    auto volume_renderer = make_operator<holoscan::ops::VolumeRendererOp>(
        "volume_renderer",
        holoscan::Arg("config_file", render_config_file_),
        holoscan::Arg("write_config_file", write_config_file_));

    auto convert_depth =
        make_operator<holoscan::openxr::ConvertDepthToScreenSpaceOp>("convert_depth");

    // OpenXR render loop.
    add_flow(xr_begin_frame, xr_end_frame, {{"xr_frame", "xr_frame"}});

    // volume data loader
    add_flow(density_volume_loader,
             volume_renderer,
             {
                 {"volume", "density_volume"},
                 {"spacing", "density_spacing"},
                 {"permute_axis", "density_permute_axis"},
                 {"flip_axes", "density_flip_axes"},
             });
    add_flow(density_volume_loader, xr_transform_controller, {{"extent", "extent"}});

    if (mask_volume_loader) {
      add_flow(mask_volume_loader,
               volume_renderer,
               {
                   {"volume", "mask_volume"},
                   {"spacing", "mask_spacing"},
                   {"permute_axis", "mask_permute_axis"},
                   {"flip_axes", "mask_flip_axes"},
               });
    }

    // Transform the volume with controller input.
    add_flow(xr_begin_frame,
             xr_transform_controller,
             {
                 {"aim_pose", "aim_pose"},
                 {"head_pose", "head_pose"},
                 {"shoulder_click", "shoulder_click"},
                 {"trigger_click", "trigger_click"},
                 {"trackpad", "trackpad"},
                 {"trackpad_touch", "trackpad_touch"},
             });

    add_flow(xr_begin_frame,
             xr_transform_renderer,
             {
                 {"depth_range", "depth_range"},
                 {"left_camera_pose", "left_camera_pose"},
                 {"right_camera_pose", "right_camera_pose"},
                 {"left_camera_model", "left_camera_model"},
                 {"right_camera_model", "right_camera_model"},
             });

    add_flow(xr_begin_frame,
             volume_renderer,
             {
                 {"depth_range", "depth_range"},
                 {"left_camera_pose", "left_camera_pose"},
                 {"right_camera_pose", "right_camera_pose"},
                 {"left_camera_model", "left_camera_model"},
                 {"right_camera_model", "right_camera_model"},
                 {"eye_gaze_pose", "eye_gaze_pose"},
             });

    add_flow(xr_transform_controller,
             volume_renderer,
             {
                 {"crop_box", "crop_box"},
                 {"volume_pose", "volume_pose"},
             });

    add_flow(xr_transform_controller,
             xr_transform_renderer,
             {{"ux_box", "ux_box"}, {"ux_cursor", "ux_cursor"}, {"ux_window", "ux_window"}});

#define SOURCE_SINK 1

#ifdef SOURCE_SINK
    auto volume_pose = std::make_shared<nlohmann::json>();
    auto render_settings_source = make_operator<holoscan::ops::SharedDataSourceOp<nlohmann::json>>(
        "render_settings_source", holoscan::Arg("shared", volume_pose));
    auto render_settings_sink = make_operator<holoscan::ops::SharedDataSinkOp<nlohmann::json>>(
        "render_settings_sink", holoscan::Arg("shared", volume_pose));

    add_flow(xr_transform_renderer,
             render_settings_sink,
             {
                 {"render_settings", "in"},
             });

    add_flow(render_settings_source,
             volume_renderer,
             {
                 {"out", "merge_settings"},
             });
#else

    add_flow(xr_transform_renderer,
             volume_renderer,
             {
                 {"render_settings", "merge_settings"},
             });

#endif
    add_flow(xr_begin_frame, volume_renderer, {{"color_buffer", "color_buffer_in"}});
    add_flow(volume_renderer, xr_transform_renderer, {{"color_buffer_out", "color_buffer_in"}});
    add_flow(xr_transform_renderer, xr_end_frame, {{"color_buffer_out", "color_buffer"}});

    add_flow(xr_begin_frame, volume_renderer, {{"depth_buffer", "depth_buffer_in"}});
    add_flow(volume_renderer, convert_depth, {{"depth_buffer_out", "depth_buffer_in"}});
    add_flow(xr_begin_frame, convert_depth, {{"depth_range", "depth_range"}});
    add_flow(convert_depth, xr_transform_renderer, {{"depth_buffer_out", "depth_buffer_in"}});
    add_flow(xr_transform_renderer, xr_end_frame, {{"depth_buffer_out", "depth_buffer"}});
  }

  const std::string render_config_file_;
  const std::string write_config_file_;
  const std::string density_volume_file_;
  const std::string mask_volume_file_;
  const bool enable_eye_tracking_;
};

int main(int argc, char** argv) {
  // Default paths in the HoloHub development container
  const std::string render_config_file_default(
      "/workspace/holohub/applications/volume_rendering_xr/configs/ctnv_bb_er.json");
  const std::string density_volume_file_default(
      "/workspace/holohub/data/volume_rendering_xr/highResCT.mhd");
  const std::string mask_volume_file_default(
      "/workspace/holohub/data/volume_rendering_xr/smoothmasks.seg.mhd");

  std::string render_config_file(render_config_file_default);
  std::string write_config_file;
  std::string density_volume_file;
  std::string mask_volume_file;
  bool enable_eye_tracking = false;

  struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                  {"config", required_argument, 0, 'c'},
                                  {"write_config", required_argument, 0, 'w'},
                                  {"density", required_argument, 0, 'd'},
                                  {"mask", required_argument, 0, 'm'},
                                  {"eye-tracking", no_argument, 0, 'e'},
                                  {0, 0, 0, 0}};

  // parse options
  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "hc:w:d:m:e", long_options, &option_index);

    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        std::cout
            << "Holoscan OpenXR volume renderer." << "Usage: " << argv[0] << " [options]"
            << std::endl
            << "Options:" << std::endl
            << "  -h, --help                            Display this information" << std::endl
            << "  -c <FILENAME>, --config <FILENAME>    Name of the renderer JSON "
               "configuration file to load (default '"
            << render_config_file_default << "')" << std::endl
            << "  -w <FILENAME>, --write_config <FILENAME> Name of the renderer JSON "
               "configuration file to write to (default '')"
            << std::endl
            << "  -d <FILENAME>, --density <FILENAME>   Name of density volume file to load "
               "(default '"
            << density_volume_file_default << "')" << std::endl
            << "  -m <FILENAME>, --mask <FILENAME>      Name of mask volume file to load "
               "(default '"
            << mask_volume_file_default << "')" << std::endl
            << "  -e, --eye-tracking                    Enable eye tracking and foveated rendering."
            << std::endl;
        return 0;

      case 'c':
        render_config_file = argument;
        break;

      case 'w':
        write_config_file = argument;
        break;

      case 'd':
        density_volume_file = argument;
        break;

      case 'm':
        mask_volume_file = argument;
        break;

      case 'e':
        enable_eye_tracking = true;
        break;

      case '?':
        // unknown option, error already printed by getop_long
        break;
      default:
        holoscan::log_error("Unhandled option '{}'", static_cast<char>(c));
    }
  }

  if (density_volume_file.empty()) {
    density_volume_file = density_volume_file_default;
    mask_volume_file = mask_volume_file_default;
  }

  App app(render_config_file,
          write_config_file,
          density_volume_file,
          mask_volume_file,
          enable_eye_tracking);
  app.run();
  return 0;
}
