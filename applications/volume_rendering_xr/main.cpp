/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "holoscan/holoscan.hpp"
#include "holoscan/operators/holoviz/holoviz.hpp"
#include "convert_depth_to_screen_space_op.hpp"
#include "xr_begin_frame_op.hpp"
#include "xr_end_frame_op.hpp"
#include "xr_transform_control_op.hpp"
#include "xr_transform_render_op.hpp"

#include "volume_loader.hpp"
#include "volume_renderer.hpp"

#include <string>

#include <getopt.h>

class App : public holoscan::Application {
 public:
  App(const std::string& render_config_file, const std::string& density_volume_file,
      const std::string& mask_volume_file)
      : render_config_file_(render_config_file),
        density_volume_file_(density_volume_file),
        mask_volume_file_(mask_volume_file) {}
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
        "xr_begin_frame", holoscan::Arg("session") = xr_session);
    auto xr_end_frame = make_operator<holoscan::openxr::XrEndFrameOp>(
        "xr_end_frame", holoscan::Arg("session") = xr_session);

    auto xr_transform_controller =
        make_operator<holoscan::openxr::XrTransformControlOp>("xr_transform_controller");
    auto xr_transform_renderer = make_operator<holoscan::openxr::XrTransformRenderOp>(
        "xr_transform_render",
        holoscan::Arg("display_width", xr_session->display_width()),
        holoscan::Arg("display_height", xr_session->display_height()));

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
        "volume_renderer", holoscan::Arg("config_file", render_config_file_));

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
             });

    add_flow(xr_transform_controller,
             volume_renderer,
             {
                 {"crop_box", "crop_box"},
                 {"volume_pose", "volume_pose"},
             });

    add_flow(xr_transform_controller,
             xr_transform_renderer,
             {
                 {"ux_box", "ux_box"},
                 {"ux_cursor", "ux_cursor"},
             });

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
  const std::string density_volume_file_;
  const std::string mask_volume_file_;
};

int main(int argc, char** argv) {
  const std::string render_config_file_default(
      "/workspace/holoscan-openxr/data/volume_rendering/config.json");
  const std::string density_volume_file_default(
      "/workspace/holoscan-openxr/data/volume_rendering/highResCT.mhd");
  const std::string mask_volume_file_default(
      "/workspace/holoscan-openxr/data/volume_rendering/smoothmasks.seg.mhd");

  std::string render_config_file;
  std::string density_volume_file;
  std::string mask_volume_file;

  struct option long_options[] = {{"help", no_argument, 0, 'h'},
                                  {"config", required_argument, 0, 'c'},
                                  {"density", required_argument, 0, 'd'},
                                  {"mask", required_argument, 0, 'm'},
                                  {0, 0, 0, 0}};

  // parse options
  while (true) {
    int option_index = 0;

    const int c = getopt_long(argc, argv, "hc:d:m:", long_options, &option_index);

    if (c == -1) { break; }

    const std::string argument(optarg ? optarg : "");
    switch (c) {
      case 'h':
        std::cout << "Holoscan OpenXR volume renderer."
                  << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  -h, --help                            Display this information" << std::endl
                  << "  -c <FILENAME>, --config <FILENAME>    Name of the renderer JSON "
                     "configuration file to load (default '"
                  << render_config_file_default << "')" << std::endl
                  << "  -d <FILENAME>, --density <FILENAME>   Name of density volume file to load "
                     "(default '"
                  << density_volume_file_default << "')" << std::endl
                  << "  -m <FILENAME>, --mask <FILENAME>      Name of mask volume file to load "
                     "(default '"
                  << mask_volume_file_default << "')" << std::endl;
        return 0;

      case 'c':
        render_config_file = argument;
        break;

      case 'd':
        density_volume_file = argument;
        break;

      case 'm':
        mask_volume_file = argument;
        break;

      case '?':
        // unknown option, error already printed by getop_long
        break;
      default:
        holoscan::log_error("Unhandled option '{}'", static_cast<char>(c));
    }
  }

  if (render_config_file.empty()) { render_config_file = render_config_file_default; }
  if (density_volume_file.empty()) {
    density_volume_file = density_volume_file_default;
    mask_volume_file = mask_volume_file_default;
  }

  App app(render_config_file, density_volume_file, mask_volume_file);
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/app_config.yaml";
  app.config(config_path);
  app.run();
  return 0;
}
