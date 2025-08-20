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
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <videomaster_source.hpp>

#include <getopt.h>

class App : public holoscan::Application {
 public:
  /** Compose function */
  void compose() override {
    using namespace holoscan;

    uint32_t width = from_config("deltacast.width").as<uint32_t>();
    uint32_t height = from_config("deltacast.height").as<uint32_t>();
    bool use_rdma = from_config("deltacast.rdma").as<bool>();
    uint64_t source_block_size = width * height * 4 * 4;
    uint64_t source_num_blocks = use_rdma ? 3 : 4;

    // Create the VideoMaster source operator (receiver) with explicit arguments
    auto source = make_operator<ops::VideoMasterSourceOp>(
        "deltacast_source",
        Arg("rdma") = use_rdma,
        Arg("board") = from_config("deltacast.board").as<uint32_t>(),
        Arg("input") = from_config("deltacast.input").as<uint32_t>(),
        Arg("width") = width,
        Arg("height") = height,
        Arg("progressive") = from_config("deltacast.progressive").as<bool>(),
        Arg("framerate") = from_config("deltacast.framerate").as<uint32_t>(),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));

    // Format converter to prepare for visualization
    auto format_converter =
        make_operator<ops::FormatConverterOp>("format_converter",
                                              from_config("format_converter"),
                                              Arg("pool") = make_resource<BlockMemoryPool>(
                                                  "converter_pool", 1, source_block_size, source_num_blocks));

    
    auto drop_alpha_channel_converter = make_operator<ops::FormatConverterOp>(
            "drop_alpha_channel_converter",
            from_config("drop_alpha_channel_converter"),
            Arg("pool") =
                make_resource<BlockMemoryPool>("pool", 1, source_block_size, source_num_blocks));
    
    // Holoviz for visualization
    auto visualizer = make_operator<ops::HolovizOp>(
        "holoviz",
        from_config("holoviz"),
        Arg("width") = width,
        Arg("height") = height,
        Arg("allocator") = make_resource<UnboundedAllocator>("holoviz_allocator"));

    // Connect the pipeline: source -> format_converter -> holoviz
    add_flow(source, drop_alpha_channel_converter);
    add_flow(drop_alpha_channel_converter, format_converter);
    add_flow(format_converter, visualizer, {{"", "receivers"}});
  }
};

/** Helper function to parse the command line arguments */
bool parse_arguments(int argc, char** argv, std::string& config_name) {
  static struct option long_options[] = {
      {"config",  required_argument, 0,  'c' },
      {"help",    no_argument,       0,  'h' },
      {0,         0,                 0,  0 }
  };

  while (int c = getopt_long(argc, argv, "c:h",
                   long_options, NULL))  {
    if (c == -1 || c == '?') break;

    switch (c) {
      case 'c':
        config_name = optarg;
        break;
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options] [config_file]\n";
        std::cout << "Options:\n";
        std::cout << "  -c, --config <file>    Configuration file path\n";
        std::cout << "  -h, --help             Show this help message\n";
        std::cout << "\nExample:\n";
        std::cout << "  " << argv[0] << " deltacast_receiver.yaml\n";
        return false;
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
  std::string config_name = "";
  if (!parse_arguments(argc, argv, config_name)) {
    return 1;
  }

  if (config_name != "") {
    app->config(config_name);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/deltacast_receiver.yaml";
    app->config(config_path);
  }

  app->run();

  return 0;
}
