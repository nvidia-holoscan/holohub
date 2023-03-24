/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 DELTACAST.TV. All rights reserved.
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
#include <holoscan/std_ops.hpp>
#include <videomaster_transmitter.hpp>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    uint32_t width = from_config("videomaster.width").as<uint32_t>();
    uint32_t height = from_config("videomaster.height").as<uint32_t>();
    uint64_t source_block_size = width * height * 4 * 4;
    uint64_t source_num_blocks = from_config("videomaster.use_rdma").as<bool>() ? 3 : 4;

    auto source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"));

    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              from_config("output_format_converter"),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks));

    auto visualizer = make_operator<ops::VideoMasterTransmitterOp>(
        "videomaster",
        from_config("videomaster"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));

    add_flow(source, format_converter);
    add_flow(format_converter, visualizer);
  }
};

int main(int argc, char** argv) {
  holoscan::load_env_log_level();

  auto app = holoscan::make_application<App>();

  if (argc == 2) {
    app->config(argv[1]);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/app_config.yaml";
    app->config(config_path);
  }
  app->run();

  return 0;
}
