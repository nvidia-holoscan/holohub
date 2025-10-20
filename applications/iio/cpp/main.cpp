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

#include <holoscan/core/conditions/gxf/boolean.hpp>
#include <holoscan/core/conditions/gxf/count.hpp>
#include <holoscan/core/endpoint.hpp>
#include <holoscan/core/forward_def.hpp>
#include <holoscan/core/operator.hpp>
#include "holoscan/holoscan.hpp"
#include "pluto_fft_example/pluto_fft_example.hpp"

// Forward declaration for basic examples main
int basic_examples_main(int argc, char** argv);

int main(int argc, char** argv) {
  // Check command line arguments
  bool realtime = false;
  bool basic_examples = false;
  int basic_example_start_index = -1;

  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--realtime") {
      realtime = true;
    } else if (std::string(argv[i]) == "--basic-example") {
      basic_examples = true;
      basic_example_start_index = i;
      break;  // Stop parsing here, let basic_examples_main handle the rest
    } else if (std::string(argv[i]) == "--help" || std::string(argv[i]) == "-h") {
      std::cout << "Usage: " << argv[0] << " [options]\n";
      std::cout << "Options:\n";
      std::cout << "  --realtime         Run Pluto FFT example in real-time mode\n";
      std::cout << "  --basic-example [basic-options]  Run basic IIO example\n";
      std::cout << "  --help, -h         Show this help message\n";
      std::cout << "\nDefault: Run Pluto FFT example in single-shot mode\n";
      std::cout << "\nBasic example options (use after --basic-example):\n";
      std::cout << "  -r, --attr-read     Run attribute read example\n";
      std::cout << "  -w, --attr-write    Run attribute write example\n";
      std::cout << "  -b, --buffer-read   Run buffer read example (default)\n";
      std::cout << "  -B, --buffer-write  Run buffer write example\n";
      std::cout << "  -c, --configurator  Run configurator example\n";
      return 0;
    }
  }

  if (basic_examples) {
    // Create new argv array starting from --basic-example
    int new_argc = argc - basic_example_start_index;
    char** new_argv = argv + basic_example_start_index;
    new_argv[0] = argv[0];  // Keep the program name

    basic_examples_main(new_argc, new_argv);
  } else if (realtime) {
    pluto_fft_realtime_main(argc, argv);
  } else {
    pluto_fft_main(argc, argv);
  }

  return 0;
}
