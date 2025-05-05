/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 DELTACAST.TV. All rights reserved.
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
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <videomaster_transmitter.hpp>

#include <getopt.h>

class App : public holoscan::Application {
 public:
  /** Sets the path to the data directory */
  void set_datapath(const std::string& path) {
     datapath = path;
  }

  /** Compose function */
  void compose() override {
    using namespace holoscan;

    uint32_t width = from_config("deltacast.width").as<uint32_t>();
    uint32_t height = from_config("deltacast.height").as<uint32_t>();
    uint64_t source_block_size = width * height * 4 * 4;
    uint64_t source_num_blocks = from_config("deltacast.rdma").as<bool>() ? 3 : 4;

    auto source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"),
                                                            Arg("directory", datapath));

    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              from_config("output_format_converter"),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "pool", 1, source_block_size, source_num_blocks));

    auto visualizer = make_operator<ops::VideoMasterTransmitterOp>(
        "deltacast",
        from_config("deltacast"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));

    add_flow(source, format_converter);
    add_flow(format_converter, visualizer);
  }

 private:
  std::string datapath = "data/endoscopy";
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name, std::string& data_path) {
  static struct option long_options[] = {
      {"data",    required_argument, 0,  'd' },
      {0,         0,                 0,  0 }
  };

  while (int c = getopt_long(argc, argv, "d",
                   long_options, NULL))  {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'd':
        data_path = optarg;
        break;
      default:
        std::cout << "Unknown arguments returned: " << c << std::endl;
        return false;
    }
  }

  if (optind < argc) {
    config_name = argv[optind++];
  }
  return true;
}

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Parse the arguments
  std::string data_path = "";
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name, data_path)) {
    return 1;
  }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/deltacast_transmitter.yaml";
    app->config(config_path);
  }

  if (data_path != "") app->set_datapath(data_path);

  app->run();

  return 0;
}
