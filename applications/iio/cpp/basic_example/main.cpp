/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Analog Devices, Inc. All rights reserved.
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

#include <yaml-cpp/exceptions.h>
#include <filesystem>
#include <holoscan/core/conditions/gxf/boolean.hpp>
#include <holoscan/core/conditions/gxf/count.hpp>
#include <holoscan/core/endpoint.hpp>
#include <holoscan/core/forward_def.hpp>
#include <holoscan/core/operator.hpp>
#include "holoscan/holoscan.hpp"
#include "iio_attribute_read.hpp"
#include "iio_attribute_write.hpp"
#include "iio_buffer_read.hpp"
#include "iio_buffer_write.hpp"
#include "iio_configurator.hpp"
#include "iio_params.hpp"
#include "support_operators.hpp"

#include <dlpack/dlpack.h>
#include <getopt.h>
#include <iostream>

class IIOBasicExampleApp : public holoscan::Application {
 public:
  IIOBasicExampleApp() { name_ = "IIO Basic Examples"; }

  void set_args(int argc, char** argv) {
    argc_ = argc;
    argv_ = argv;
  }

  void attr_read_example() {
    using namespace holoscan;
    HOLOSCAN_LOG_INFO("Setting up attribute read example");

    auto iio_read_cond = make_condition<CountCondition>("iio_read_cond", G_NUM_READS);

    auto iio_read_op =
        make_operator<ops::IIOAttributeRead>("iio_attribute_read",
                                             Arg("ctx") = std::string(G_URI),
                                             Arg("dev") = std::string("ad9361-phy"),
                                             Arg("attr_name") = std::string("trx_rate_governor"),
                                             iio_read_cond);

    auto basic_print_op = make_operator<ops::BasicPrinterOp>("basic_print_op");

    add_flow(iio_read_op, basic_print_op, {{"value", "value"}});
  }

  void attr_write_example() {
    using namespace holoscan;
    HOLOSCAN_LOG_INFO("Setting up attribute write example");

    auto iio_write_cond = make_condition<CountCondition>("iio_write_cond", G_NUM_READS);

    auto iio_write_op =
        make_operator<ops::IIOAttributeWrite>("iio_attribute_write",
                                              Arg("ctx") = std::string(G_URI),
                                              Arg("dev") = std::string("ad9361-phy"),
                                              Arg("attr_name") = std::string("trx_rate_governor"),
                                              iio_write_cond);

    auto basic_emit_op = make_operator<ops::BasicEmitterOp>("basic_emit_op");

    add_flow(basic_emit_op, iio_write_op, {{"value", "value"}});
  }

  void buffer_read_example() {
    using namespace holoscan;
    HOLOSCAN_LOG_INFO("Setting up buffer read example");

    auto iio_rw_cond = make_condition<CountCondition>("iio_read_cond", 1);

    // Channel configuration based on G_NUM_CHANNELS
    std::vector<std::string> enabled_channels_names_1;
    std::vector<bool> enabled_channels_output;

    enabled_channels_names_1.push_back("voltage0");
    enabled_channels_output.push_back(false);  // False for input channels

    if (G_NUM_CHANNELS == 2) {
      enabled_channels_names_1.push_back("voltage1");
      enabled_channels_output.push_back(false);
    }

    auto iio_buf_read_op =
        make_operator<ops::IIOBufferRead>("iio_buffer_read",
                                          Arg("ctx") = std::string(G_URI),
                                          Arg("dev") = std::string("cf-ad9361-lpc"),
                                          Arg("is_cyclic") = true,
                                          Arg("samples_count") = static_cast<size_t>(8192),
                                          Arg("enabled_channel_names") = enabled_channels_names_1,
                                          Arg("enabled_channel_output") = enabled_channels_output,
                                          iio_rw_cond);

    auto basic_buffer_printer_op =
        make_operator<ops::BasicIIOBufferPrinterOP>("basic_buffer_printer_op");

    // RX flow - connect buffer reader to buffer printer
    add_flow(iio_buf_read_op, basic_buffer_printer_op, {{"buffer", "buffer"}});
  }

  void buffer_write_example() {
    using namespace holoscan;
    HOLOSCAN_LOG_INFO("Setting up buffer write example");

    auto iio_rw_cond = make_condition<CountCondition>("iio_write_cond", 1);

    // Channel configuration based on G_NUM_CHANNELS
    std::vector<std::string> enabled_channels_names_1;
    std::vector<bool> enabled_channels_output;

    enabled_channels_names_1.push_back("voltage0");
    enabled_channels_output.push_back(true);  // True for output channels

    if (G_NUM_CHANNELS == 2) {
      enabled_channels_names_1.push_back("voltage1");
      enabled_channels_output.push_back(true);
    }

    auto iio_buf_write_op_1 =
        make_operator<ops::IIOBufferWrite>("iio_buffer_write_1",
                                           Arg("ctx") = std::string(G_URI),
                                           Arg("dev") = std::string("cf-ad9361-dds-core-lpc"),
                                           Arg("is_cyclic") = true,
                                           Arg("enabled_channel_names") = enabled_channels_names_1,
                                           Arg("enabled_channel_output") = enabled_channels_output,
                                           iio_rw_cond);

    auto basic_buffer_emitter_op =
        make_operator<ops::BasicIIOBufferEmitterOP>("basic_buffer_emitter_op");

    auto basic_wait_op = make_operator<ops::BasicWaitOp>("basic_wait_op");

    // TX flow - connect buffer emitter to buffer writer to wait
    add_flow(basic_buffer_emitter_op, iio_buf_write_op_1, {{"buffer", "buffer"}});
    add_flow(iio_buf_write_op_1, basic_wait_op);
  }

  void configurator_example() {
    using namespace holoscan;
    HOLOSCAN_LOG_INFO("Setting up configurator example");

    auto config_file_path = config().config_file();
    HOLOSCAN_LOG_INFO("Config file: {}", config_file_path);

    auto iio_configurator_op = make_operator<ops::IIOConfigurator>(
        "iio_configurator_op", Arg("cfg") = std::string(config_file_path));

    // start_op() will only run the configurator once
    add_flow(start_op(), iio_configurator_op);
  }

  void compose() override {
    HOLOSCAN_LOG_INFO("IIO Basic Examples started");

    int opt;
    int example_type = 0;  // Default to buffer read

    static struct option long_options[] = {
        {"attr-read", no_argument, 0, 'r'},
        {"attr-write", no_argument, 0, 'w'},
        {"buffer-read", no_argument, 0, 'b'},
        {"buffer-write", no_argument, 0, 'B'},
        {"configurator", no_argument, 0, 'c'},
        {"help", no_argument, 0, 'h'},
        {0, 0, 0, 0}
    };

    int option_index = 0;
    optind = 1;  // Reset getopt parsing

    while ((opt = getopt_long(argc_, argv_, "rwbBch", long_options, &option_index)) != -1) {
        switch (opt) {
            case 'r':
                example_type = 1;
                break;
            case 'w':
                example_type = 2;
                break;
            case 'b':
                example_type = 3;
                break;
            case 'B':
                example_type = 4;
                break;
            case 'c':
                example_type = 5;
                break;
            case 'h':
                std::cout << "Usage: " << argv_[0] << " [options]\n"
                          << "Options:\n"
                          << "  -r, --attr-read     Run attribute read example\n"
                          << "  -w, --attr-write    Run attribute write example\n"
                          << "  -b, --buffer-read   Run buffer read example (default)\n"
                          << "  -B, --buffer-write  Run buffer write example\n"
                          << "  -c, --configurator  Run configurator example\n"
                          << "  -h, --help          Show this help message\n";
                exit(0);
            default:
                std::cerr << "Unknown option. Use -h for help.\n";
                exit(1);
        }
    }

    // Run the selected example
    switch (example_type) {
        case 1:
            attr_read_example();
            break;
        case 2:
            attr_write_example();
            break;
        case 4:
            buffer_write_example();
            break;
        case 5:
            configurator_example();
            break;
        case 3:
        default:
            buffer_read_example();
            break;
    }
  }

 private:
  std::string name_;
  int argc_ = 0;
  char** argv_ = nullptr;
};

int basic_examples_main(int argc, char** argv) {
  auto app = holoscan::make_application<IIOBasicExampleApp>();
  app->set_args(argc, argv);
  app->run();

  return 0;
}
