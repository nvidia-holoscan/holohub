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

#include "holoscan/holoscan.hpp"

#include "xr_basic_render_op.hpp"
#include "xr_begin_frame_op.hpp"
#include "xr_end_frame_op.hpp"
#include "xr_transform_control_op.hpp"
#include "xr_transform_render_op.hpp"

#include <string>

/**
 * @brief Example application demonstrating XR scene viewing with Holoscan SDK.
 *
 * @details The Hello Holoscan XR application is an example application that demonstrates the usage
 * of the Holoscan library for rendering XR content using the OpenXR API. It creates an XR session,
 * initializes the session, and sets up the rendering pipeline using various operators and resources
 * provided in HoloHub and built on Holoscan SDK.
 */
class App final : public holoscan::Application {
 public:
  App() = default;

  void compose() override {
    std::shared_ptr<holoscan::openxr::XrSession> xr_session =
        make_resource<holoscan::openxr::XrSession>(
            "xr_session",
            holoscan::Arg("application_name") = std::string("Hello Holoscan XR"),
            holoscan::Arg("application_version") = 1u,
            holoscan::Arg("near_z", 0.33f),
            holoscan::Arg("far_z", 10.f));
    // resources are lazy initialized by Holoscan but we need the session initialized here to get
    // the display size
    xr_session->initialize();

    auto xr_begin_frame = make_operator<holoscan::openxr::XrBeginFrameOp>(
        "xr_begin_frame",
        holoscan::Arg("session") = xr_session,
        holoscan::Arg("enable_eye_tracking") = true);
    auto xr_end_frame = make_operator<holoscan::openxr::XrEndFrameOp>(
        "xr_end_frame", holoscan::Arg("session") = xr_session);

    auto basic_renderer = make_operator<holoscan::ops::BasicRenderOp>(
        "basic_renderer",
        holoscan::Arg("display_width", xr_session->display_width()),
        holoscan::Arg("display_height", xr_session->display_height()));

    // OpenXR render loop.
    add_flow(xr_begin_frame, xr_end_frame, {{"xr_frame", "xr_frame"}});

    add_flow(xr_begin_frame,
             basic_renderer,
             {
                 {"depth_range", "depth_range"},
                 {"left_camera_pose", "left_camera_pose"},
                 {"right_camera_pose", "right_camera_pose"},
                 {"left_camera_model", "left_camera_model"},
                 {"right_camera_model", "right_camera_model"},
                 // inputs.
                 {"head_pose", "head_pose"},
                 {"eye_gaze_pose", "eye_gaze_pose"},
                 {"aim_pose", "aim_pose"},
                 {"grip_pose", "grip_pose"},
                 {"shoulder_click", "shoulder_click"},
                 {"trigger_click", "trigger_click"},
                 {"trackpad", "trackpad"},
                 {"trackpad_touch", "trackpad_touch"},
             });

    add_flow(xr_begin_frame, basic_renderer, {{"color_buffer", "color_buffer_in"}});
    add_flow(basic_renderer, xr_end_frame, {{"color_buffer_out", "color_buffer"}});

    add_flow(xr_begin_frame, basic_renderer, {{"depth_buffer", "depth_buffer_in"}});
    add_flow(basic_renderer, xr_end_frame, {{"depth_buffer_out", "depth_buffer"}});
  }
};

int main(int argc, char** argv) {
  App app;
  app.run();
  return 0;
}
