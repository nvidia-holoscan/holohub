/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/inference_processor/inference_processor.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>


class SrDemoApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;
    auto replayer =
        make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"), 
        Arg("allocator") = make_resource<UnboundedAllocator>("pool_replayer"));
    auto visualizer = make_operator<ops::HolovizOp>("holoviz", from_config("holoviz"));


    auto inference = make_operator<ops::InferenceOp>(
        "inference",
        from_config("sr_inference"),
        Arg("allocator") = make_resource<UnboundedAllocator>("pool_inference"));


      auto drop_alpha =
        make_operator<ops::FormatConverterOp>("drop_alpha", from_config("drop_alpha"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool_drop_alpha"));

      auto preprocessor =
        make_operator<ops::FormatConverterOp>("preprocessor", from_config("inference_preprocessor"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool_preprocessor"));

     auto postprocessor =
        make_operator<ops::FormatConverterOp>("postprocessor", from_config("inference_postprocessor"),
         Arg("pool") = make_resource<UnboundedAllocator>("pool_postprocessor"));


    add_flow(replayer, drop_alpha);
    add_flow(drop_alpha, preprocessor);
    add_flow(preprocessor, inference);
    add_flow(inference, postprocessor);
    add_flow(postprocessor, visualizer, {{"tensor", "receivers"}});
 
  }
};

int main(int argc, char** argv) {

    int opt;
    std::string config_file;
    bool tracking = false;
    
    auto default_path = std::filesystem::canonical(argv[0]).parent_path();
    default_path /= std::filesystem::path("sr_demo.yaml");

    while ((opt = getopt(argc, argv, "c:t")) != -1) {
        switch (opt) {
            case 'c':
                config_file = optarg;
                break;
            case 't':
                tracking = true;
                break;
            default:
                std::cerr << "Usage: " << argv[0] << " -c configuration_file [-t]" << std::endl;
                return EXIT_FAILURE;
        }
    }

    if (config_file.empty()) {
      config_file = default_path.string();
    }

  auto app = holoscan::make_application<SrDemoApp>();
  if(tracking){
    auto& track = app->track();
    track.enable_logging("super_resolution_demo.log");
  }
  
  app->config(config_file);
  app->run();

  return 0;
}

