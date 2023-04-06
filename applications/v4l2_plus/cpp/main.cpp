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
#include <v4l2_plus_source.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto source = make_operator<ops::V4L2PlusSourceOp>(
        "source",
        from_config("source"),
        Arg("allocator") = make_resource<UnboundedAllocator>("allocator"));

    auto sink = make_operator<ops::HolovizOp>(
        "sink",
        from_config("sink"));

    // Flow definition
    add_flow(source, sink, {{"signal", "receivers"}});
  }
};

int main(int argc, char** argv) {
  App app;

  // Get the configuration
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/v4l2_plus.yaml";
  app.config(config_path);

  app.run();

  return 0;
}

