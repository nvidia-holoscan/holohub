/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <emergent_source.hpp>
#include <holoscan/operators/bayer_demosaic/bayer_demosaic.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    // Create instances of all the operators being used
    std::shared_ptr<Operator> source;
    std::shared_ptr<Operator> bayer_demosaic;
    std::shared_ptr<Operator> viz;

    // emergent camera is the source for this app for data acquisition in Bayer format
    source = make_operator<ops::EmergentSourceOp>("emergent", from_config("emergent"));

    auto cuda_stream_pool = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

// Fix to initialize the cuda_stream_pool for Holoscan 0.6 only
#if HOLOSCAN_MAJOR_VERSION == 0 && HOLOSCAN_MINOR_VERSION == 6
    cuda_stream_pool->initialize();
#endif

    // bayer demosaic is the post processing step to convert Bayer frame to RGB format
    bayer_demosaic = make_operator<ops::BayerDemosaicOp>(
        "bayer_demosaic",
        from_config("demosaic"),
        Arg("pool") = make_resource<BlockMemoryPool>("pool", 1, 72576000, 2),
        Arg("cuda_stream_pool") = cuda_stream_pool);

    // Holoviz is the visualizer being used for the peak performance
    viz = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));

    // Create the pipeline source->bayer_demosaic->viz
    add_flow(source, bayer_demosaic, {{"signal", "receiver"}});
    add_flow(bayer_demosaic, viz, {{"transmitter", "receivers"}});
  }

 private:
};

int main(int argc, char** argv) {
  // Create an instance of App
  auto app = holoscan::make_application<App>();

  // Read in the parameters provided at the command line
  if (argc == 2) {
    app->config(argv[1]);
  } else {
    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/high_speed_endoscopy.yaml";
    app->config(config_path);
  }

  // Run the App
  app->run();

  return 0;
}
