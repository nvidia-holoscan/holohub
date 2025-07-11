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

#include <map>
#include <random>
#include <cmath>
#include <chrono>
#include <thread>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/operators/format_converter/format_converter.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/segmentation_postprocessor/segmentation_postprocessor.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include <openigtlink_tx.hpp>
#include <openigtlink_rx.hpp>

class OpenIGTLinkApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    const std::shared_ptr<CudaStreamPool> cuda_stream_pool =
        make_resource<CudaStreamPool>("cuda_stream", 0, 0, 0, 1, 5);

    // VideoStreamReplayerOp
    auto replayer = make_operator<ops::VideoStreamReplayerOp>(
      "replayer",
      from_config("replayer"),
      Arg("directory", datapath_));

    // OpenIGTLinkTxOp
    auto openigtlink_tx_slicer_img = make_operator<ops::OpenIGTLinkTxOp>(
      "openigtlink_tx_slicer_img",
      from_config("openigtlink_tx_slicer_img"));

    // OpenIGTLinkRxOp
    auto openigtlink_rx_slicer_img = make_operator<ops::OpenIGTLinkRxOp>(
      "openigtlink_rx_slicer_img",
      from_config("openigtlink_rx_slicer_img"),
      Arg("allocator") = make_resource<UnboundedAllocator>("pool"));

    // FormatConverterOp
    const int n_channels = 3;
    const int width_preprocessor = 256;
    const int height_preprocessor = 256;
    uint64_t preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * 1;
    uint64_t preprocessor_num_blocks = 2;
    auto uint8_preprocessor = make_operator<ops::FormatConverterOp>(
      "uint8_preprocessor",
      from_config("uint8_preprocessor"),
      Arg("in_tensor_name", std::string("")),
      Arg("pool") = make_resource<BlockMemoryPool>(
          "pool", 1, preprocessor_block_size, preprocessor_num_blocks),
      Arg("cuda_stream_pool") = cuda_stream_pool);

    // FormatConverterOp
    preprocessor_block_size = width_preprocessor * height_preprocessor * n_channels * 4;
    auto segmentation_preprocessor = make_operator<ops::FormatConverterOp>(
      "segmentation_preprocessor",
      from_config("segmentation_preprocessor"),
      Arg("in_tensor_name", std::string("")),
      Arg("pool") = make_resource<BlockMemoryPool>(
          "pool", 1, preprocessor_block_size, preprocessor_num_blocks),
      Arg("cuda_stream_pool") = cuda_stream_pool);

    // InferenceOp
    const int n_channels_inference = 2;
    const int width_inference = 256;
    const int height_inference = 256;
    const int bpp_inference = 4;
    const uint64_t inference_block_size =
        width_inference * height_inference * n_channels_inference * bpp_inference;
    const uint64_t inference_num_blocks = 2;
    ops::InferenceOp::DataMap model_path_map;
    model_path_map.insert("ultrasound_seg", datapath_ + "/us_unet_256x256_nhwc.onnx");
    auto segmentation_inference = make_operator<ops::InferenceOp>(
        "segmentation_inference_holoinfer",
        from_config("segmentation_inference_holoinfer"),
        Arg("model_path_map", model_path_map),
        Arg("allocator") =
            make_resource<BlockMemoryPool>("pool", 1, inference_block_size, inference_num_blocks));

    // SegmentationPostprocessorOp
    const uint64_t postprocessor_block_size = width_inference * height_inference;
    const uint64_t postprocessor_num_blocks = 2;
    auto segmentation_postprocessor = make_operator<ops::SegmentationPostprocessorOp>(
      "segmentation_postprocessor",
      from_config("segmentation_postprocessor"),
      Arg("allocator") = make_resource<BlockMemoryPool>(
          "pool", 1, postprocessor_block_size, postprocessor_num_blocks));

    // HolovizOp
    auto segmentation_visualizer = make_operator<ops::HolovizOp>(
      "segmentation_visualizer",
      from_config("segmentation_visualizer"),
      Arg("allocator") = make_resource<UnboundedAllocator>("pool"),
      Arg("cuda_stream_pool") = cuda_stream_pool);

    // OpenIGTLinkTxOp
    auto openigtlink_tx_slicer_holoscan = make_operator<ops::OpenIGTLinkTxOp>(
      "openigtlink_tx_slicer_holoscan",
      from_config("openigtlink_tx_slicer_holoscan"));

    // Build flow
    add_flow(replayer, uint8_preprocessor, {{"", "source_video"}});
    add_flow(uint8_preprocessor, openigtlink_tx_slicer_img, {{"tensor", "receivers"}});
    add_flow(openigtlink_rx_slicer_img, segmentation_visualizer, {{"out_tensor", "receivers"}});
    add_flow(openigtlink_rx_slicer_img,
             segmentation_preprocessor, {{"out_tensor", "source_video"}});
    add_flow(segmentation_preprocessor, segmentation_inference, {{"tensor", "receivers"}});
    add_flow(segmentation_inference, segmentation_postprocessor, {{"transmitter", ""}});
    add_flow(segmentation_postprocessor, segmentation_visualizer, {{"", "receivers"}});
    add_flow(segmentation_visualizer,
             openigtlink_tx_slicer_holoscan, {{"render_buffer_output", "receivers"}});
  }

 private:
  std::string datapath_ = "/workspace/holohub/data/ultrasound_segmentation";
};

int main(int argc, char** argv) {
  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path /= std::filesystem::path("openigtlink_3dslicer.yaml");
  if ( argc >= 2 ) {
    config_path = argv[1];
  }

  auto app = holoscan::make_application<OpenIGTLinkApp>();
  app->config(config_path);
  app->run();

  return 0;
}
