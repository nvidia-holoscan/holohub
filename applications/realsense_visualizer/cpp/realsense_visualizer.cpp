/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "realsense_camera.hpp"

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto allocator = make_resource<UnboundedAllocator>("allocator");

    auto realsense_camera =
        make_operator<ops::RealsenseCameraOp>("realsense_camera", Arg("allocator") = allocator);

    auto color_visualizer =
        make_operator<ops::HolovizOp>("color_visualizer",
                                      Arg("window_title") = std::string("RealSense Color"),
                                      Arg("allocator") = allocator);

    std::vector<ops::HolovizOp::InputSpec> depth_input_spec = {
        ops::HolovizOp::InputSpec("", ops::HolovizOp::InputType::COLOR)};
    auto depth_visualizer =
        make_operator<ops::HolovizOp>("depth_visualizer",
                                      Arg("tensors") = depth_input_spec,
                                      Arg("window_title") = std::string("RealSense Depth"),
                                      Arg("allocator") = allocator);

    add_flow(realsense_camera, depth_visualizer, {{"depth_buffer", "receivers"}});
    add_flow(realsense_camera, color_visualizer, {{"color_buffer", "receivers"}});
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();
  app->run();

  return 0;
}
