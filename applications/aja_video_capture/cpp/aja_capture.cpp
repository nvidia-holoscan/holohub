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

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <aja_source.hpp>
class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto source = make_operator<ops::AJASourceOp>("aja", from_config("aja"),
                                make_condition<CountCondition>(from_config("aja.count")));
    auto visualizer = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));

    // Flow definition
    add_flow(source, visualizer, {{"video_buffer_output", "receivers"}});
  }
};

int main(int argc, char** argv) {
  App app;

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("aja_capture.yaml");
  if (argc >= 2) {
    config_path = argv[1];
  }

  app.config(config_path);

  app.run();

  return 0;
}
