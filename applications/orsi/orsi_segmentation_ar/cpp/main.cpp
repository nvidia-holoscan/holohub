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

// Holoscan SDK Operators

#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/inference/inference.hpp>
// Holohub operators

#include <segmentation_postprocessor.hpp>
#include <segmentation_preprocessor.hpp>
#include <orsi_visualizer.hpp>

#include <orsi_app.hpp>
class App : public OrsiApp {
 public:
  void compose() override {
    using namespace holoscan;

    std::shared_ptr<Resource> allocator_resource =
        make_resource<UnboundedAllocator>("unbounded_allocator");

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    initVideoSource(cuda_stream_pool);

    // -------------------------------------------------------------------------------------
    //
    // Format conversion operators
    //

    int width_preprocessor = 1264;
    int height_preprocessor = 1080;
    uint64_t preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * bpp;
    uint64_t preprocessor_num_blocks = 2;

    auto format_converter = make_operator<ops::orsi::FormatConverterOp>(
        "format_converter",
        from_config("format_converter"),
        Arg("in_tensor_name", video_format_converter_in_tensor_name),
        Arg("allocator") = allocator_resource);

    // -------------------------------------------------------------------------------------
    //
    // Pre-processing Operators
    //

    auto segmentation_preprocessor = make_operator<ops::orsi::SegmentationPreprocessorOp>(
        "segmentation_preprocessor",
        from_config("segmentation_preprocessor"),
        Arg("allocator") = allocator_resource);

    // -------------------------------------------------------------------------------------
    //
    // Multi-AI Inference Operator
    //

    ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("tool_segmentation", datapath + "/models/segmentation_model.onnx");

    auto multiai_inference = make_operator<ops::InferenceOp>("multiai_inference",
                                                             from_config("multiai_inference"),
                                                             Arg("model_path_map", model_path_map),
                                                             Arg("allocator") = allocator_resource);

    // -------------------------------------------------------------------------------------
    //
    // Post-processing Operators
    //

    auto segmentation_postprocessor = make_operator<ops::orsi::SegmentationPostprocessorOp>(
        "segmentation_postprocessor",
        from_config("segmentation_postprocessor"),
        Arg("allocator") = allocator_resource);

    // -------------------------------------------------------------------------------------
    //
    // Visualization Operator
    //
    std::cout << datapath << std::endl;
    auto orsi_visualizer = make_operator<ops::orsi::OrsiVisualizationOp>(
        "orsi_visualizer",
        from_config("orsi_visualizer"),
        Arg("stl_file_path", datapath + "/stl/segmentation/"),
        Arg("registration_params_path", datapath + "/registration_params/segmentation_ar.txt"),
        Arg("allocator") = allocator_resource);

    // Flow definition
    switch (video_source_) {
#ifdef USE_VIDEOMASTER
      case VideoSource::VIDEOMASTER:
#endif
      case VideoSource::AJA:
        add_flow(source, orsi_visualizer, {{video_buffer_out, "receivers"}});
        add_flow(source, drop_alpha_channel, {{video_buffer_out, ""}});
        add_flow(drop_alpha_channel, format_converter);
        break;

      case VideoSource::REPLAYER:
      default:
        add_flow(source, orsi_visualizer, {{video_buffer_out, "receivers"}});
        add_flow(source, format_converter, {{video_buffer_out, "source_video"}});
        break;
    }

    add_flow(format_converter, segmentation_preprocessor);
    add_flow(segmentation_preprocessor, multiai_inference, {{"", "receivers"}});
    add_flow(multiai_inference, segmentation_postprocessor, {{"transmitter", ""}});
    add_flow(segmentation_postprocessor, orsi_visualizer, {{"", "receivers"}});
  }
};

int main(int argc, char** argv) {
  holoscan::set_log_level(holoscan::LogLevel::WARN);

  auto app = holoscan::make_application<App>();
  // Parse the arguments, set source, datapath, config file
  if (!app->init(argc, argv)) { return 1; }

  auto& tracker = app->track();
  app->run();
  std::cout << "// Application::run completed. Printing tracker results" << std::endl;
  tracker.print();

  return 0;
}
