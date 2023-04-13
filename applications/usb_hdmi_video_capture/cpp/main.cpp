/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <holoscan/holoscan.hpp>
#include <v4l2_video_capture.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <typeinfo>
class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    const int width = from_config("source.width").as<int>();
    const int height = from_config("source.height").as<int>();
    const int n_channels = 4;
    uint64_t block_size = width * height * n_channels;

    auto source = make_operator<ops::V4L2VideoCaptureOp>(
      "source",
      from_config("source"),
      Arg("allocator") = make_resource<BlockMemoryPool>("pool", 0, block_size, 1)
    );

    // Set Holoviz width and height from source resolution
    auto viz_args = from_config("visualizer");
    for (auto& arg : from_config("source")) {
      if      (arg.name() == "width")  viz_args.add(arg);
      else if (arg.name() == "height") viz_args.add(arg);
    }

    auto visualizer = make_operator<ops::HolovizOp>(
      "visualizer",
      viz_args
    );

    // Flow definition
    add_flow(source, visualizer, {{"signal", "receivers"}});
  }
};

int main(int argc, char** argv) {
  App app;

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/usb_hdmi_video_capture.yaml";
  app.config(config_path);

  app.run();

  return 0;
}

