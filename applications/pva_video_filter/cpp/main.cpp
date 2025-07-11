/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "gxf/std/tensor.hpp"
#include "holoscan/holoscan.hpp"
#include "pva_unsharp_mask/pva_unsharp_mask.hpp"

#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoscan/operators/video_stream_recorder/video_stream_recorder.hpp>
#include <holoscan/operators/video_stream_replayer/video_stream_replayer.hpp>
#include <holoscan/core/system/gpu_resource_monitor.hpp>

#include <iostream>
#include <string>

namespace holoscan::ops {
class PVAVideoFilterExecutor : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(PVAVideoFilterExecutor);
  PVAVideoFilterExecutor() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_, "allocator", "Allocator", "Allocator to allocate output tensor.");
    spec.input<gxf::Entity>("input");
    spec.output<gxf::Entity>("output");
  }
  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto maybe_input_message = op_input.receive<gxf::Entity>("input");
    if (!maybe_input_message.has_value()) {
      HOLOSCAN_LOG_ERROR("Failed to receive input message gxf::Entity");
      return;
    }
    auto input_tensor = maybe_input_message.value().get<holoscan::Tensor>();
    if (!input_tensor) {
      HOLOSCAN_LOG_ERROR("Failed to receive holoscan::Tensor from input message gxf::Entity");
      return;
    }

    // get handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        fragment()->executor().context(), allocator_->gxf_cid());

    // cast Holoscan::Tensor to nvidia::gxf::Tensor to use its APIs directly
    nvidia::gxf::Tensor input_tensor_gxf{input_tensor->dl_ctx()};

    auto out_message = CreateTensorMap(
        context.context(),
        allocator.value(),
        {{"output",
          nvidia::gxf::MemoryStorageType::kDevice,
          input_tensor_gxf.shape(),
          nvidia::gxf::PrimitiveType::kUnsigned8,
          0,
          nvidia::gxf::ComputeTrivialStrides(
              input_tensor_gxf.shape(),
              nvidia::gxf::PrimitiveTypeSize(nvidia::gxf::PrimitiveType::kUnsigned8))}},
        false);

    if (!out_message) { std::runtime_error("failed to create out_message"); }
    const auto output_tensor = out_message.value().get<nvidia::gxf::Tensor>();
    if (!output_tensor) { std::runtime_error("failed to create out_tensor"); }

    uint8_t* input_tensor_data = static_cast<uint8_t*>(input_tensor->data());
    uint8_t* output_tensor_data = static_cast<uint8_t*>(output_tensor.value()->pointer());
    if (output_tensor_data == nullptr) {
      throw std::runtime_error("Failed to allocate memory for the output image");
    }

    const int32_t imageWidth{static_cast<int32_t>(input_tensor->shape()[1])};
    const int32_t imageHeight{static_cast<int32_t>(input_tensor->shape()[0])};
    const int32_t inputLinePitch{static_cast<int32_t>(input_tensor->shape()[1])};
    const int32_t outputLinePitch{static_cast<int32_t>(input_tensor->shape()[1])};

    if (!pvaOperatorTask_.isInitialized()) {
      pvaOperatorTask_.init(imageWidth, imageHeight, inputLinePitch, outputLinePitch);
    }
    pvaOperatorTask_.process(input_tensor_data, output_tensor_data);
    auto result = gxf::Entity(std::move(out_message.value()));

    op_output.emit(result, "output");
  }

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  PvaUnsharpMask pvaOperatorTask_;
};
}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    uint32_t max_width{1920};
    uint32_t max_height{1080};
    int64_t source_block_size = max_width * max_height * 3;

    std::shared_ptr<BlockMemoryPool> pva_allocator =
        make_resource<BlockMemoryPool>("allocator", 1, source_block_size, 1);

    auto pva_video_filter = make_operator<ops::PVAVideoFilterExecutor>(
        "pva_video_filter", Arg("allocator") = pva_allocator);

    auto source = make_operator<ops::VideoStreamReplayerOp>("replayer", from_config("replayer"));

    auto recorder = make_operator<ops::VideoStreamRecorderOp>("recorder", from_config("recorder"));
    auto visualizer1 = make_operator<ops::HolovizOp>(
        "holoviz1", from_config("holoviz"), Arg("window_title") = std::string("Original Stream"));
    auto visualizer2 =
        make_operator<ops::HolovizOp>("holoviz2",
                                      from_config("holoviz"),
                                      Arg("window_title") = std::string("Image Sharpened Stream"));

    add_flow(source, pva_video_filter);
    add_flow(source, visualizer1, {{"output", "receivers"}});
    // add_flow(pva_video_filter, recorder);
    add_flow(pva_video_filter, visualizer2, {{"output", "receivers"}});
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/main.yaml";
  app->config(config_path);

  app->run();

  return 0;
}
