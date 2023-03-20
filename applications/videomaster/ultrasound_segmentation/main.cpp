/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Operator> source = make_operator<ops::VideoMasterSourceOp>(
        "videomaster",
        from_config("videomaster"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"));

    const int width = 1920;
    const int height = 1080;
    const int n_channels = 4;
    const int bpp = 4;
    uint64_t drop_alpha_block_size = width * height * n_channels * bpp;
    uint64_t drop_alpha_num_blocks = 2;
    std::shared_ptr<Operator> drop_alpha_channel = make_operator<ops::FormatConverterOp>(
        "drop_alpha_channel",
        from_config("drop_alpha_channel"),
        Arg("pool") = make_resource<BlockMemoryPool>(
            "pool", 1, drop_alpha_block_size, drop_alpha_num_blocks));

    int width_preprocessor = 1264;
    int height_preprocessor = 1080;
    uint64_t preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp;
    uint64_t preprocessor_num_blocks = 2;
    auto segmentation_preprocessor = make_operator<ops::FormatConverterOp>(
        "segmentation_preprocessor",
        from_config("segmentation_preprocessor"),
        Arg("in_tensor_name", std::string("source_video")),
        Arg("pool") = make_resource<BlockMemoryPool>(
            "pool", 1, preprocessor_block_size, preprocessor_num_blocks));

    auto segmentation_inference = make_operator<ops::TensorRTInferenceOp>(
        "segmentation_inference",
        from_config("segmentation_inference"),
        Arg("pool") = make_resource<UnboundedAllocator>("pool"),
        Arg("cuda_stream_pool") = make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5));

    auto segmentation_postprocessor = make_operator<ops::SegmentationPostprocessorOp>(
        "segmentation_postprocessor",
        from_config("segmentation_postprocessor"),
        Arg("allocator") = make_resource<UnboundedAllocator>("allocator"));

    auto segmentation_visualizer = make_operator<ops::HolovizOp>(
        "segmentation_visualizer", from_config("segmentation_visualizer"));

    // Flow definition
    add_flow(source, drop_alpha_channel, {{"signal", ""}});
    add_flow(drop_alpha_channel, segmentation_visualizer, {{"", "receivers"}});
    add_flow(drop_alpha_channel, segmentation_preprocessor);

    add_flow(segmentation_preprocessor, segmentation_inference);
    add_flow(segmentation_inference, segmentation_postprocessor);
    add_flow(segmentation_postprocessor, segmentation_visualizer, {{"", "receivers"}});
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
