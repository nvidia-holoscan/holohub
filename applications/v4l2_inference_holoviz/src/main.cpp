// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <exception>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/inference/inference.hpp>
#include <holoscan/operators/tensor_preprocessor/tensor_preprocessor.hpp>
#include <holoscan/visualization/holoviz_op.hpp>
#include <v4l2_capture_op/v4l2_capture_op.hpp>

#include "options.hpp"
#include "tensor_to_image.hpp"

namespace {

int run(const camera_inference::Options& options) {
  namespace hs = holoscan;
  namespace viz = holoscan::visualization;
  if (!options.validate && !std::filesystem::is_regular_file(options.model)) {
    throw std::invalid_argument("Model not found: " + options.model +
                                ". Copy the SDK inference example identity_model.onnx "
                                "as described in the README, or pass --model.");
  }
  hs::Graph graph{"v4l2-inference-holoviz"};
  const auto camera =
      graph.op<hs::holoscan_camera::V4l2CaptureOp>("camera",
                                                   options.device,
                                                   options.width,
                                                   options.height,
                                                   options.fps,
                                                   std::string{"camera_optical_frame"},
                                                   hs::MemoryKind::kCudaDevice);

  hs::ops::TensorPreprocessorOpParams preprocessing;
  preprocessing.input_kind = "image";
  preprocessing.input_port_name = "camera_frame";
  preprocessing.output_port_name = "model_input";
  preprocessing.input_pix_fmt = "yuyv";
  preprocessing.output_width = options.model_width;
  preprocessing.output_height = options.model_height;
  preprocessing.output_pix_fmt = "rgb_fp32";
  preprocessing.output_layout = "nchw";
  preprocessing.prepend_batch_dim = true;
  preprocessing.memory_kind = hs::MemoryKind::kCudaDevice;
  const auto preprocessor = graph.op<hs::ops::TensorPreprocessorOp>("preprocessor", preprocessing);

  hs::ops::InferenceOpParams inference_params;
  inference_params.model_path_map = {{"image_model", options.model}};
  inference_params.input_map = {{"image_model", {"model_input"}}};
  inference_params.output_map = {{"image_model", {"prediction"}}};
  inference_params.parallel_inference = false;
  inference_params.input_on_cuda = true;
  inference_params.output_on_cuda = true;
  // Keep HoloInfer's output on CUDA; the display adapter downloads it for RGB conversion.
  inference_params.transmit_on_cuda = true;
  const auto inference = graph.op<hs::ops::InferenceOp>("inference", inference_params);
  const auto image = graph.op<camera_inference::TensorToImageOp>(
      "tensor_to_image", options.model_width, options.model_height);

  viz::LayerStack layers;
  layers.add_image_layer(
      "prediction",
      viz::ImageLayerDesc{.format = viz::ImageFormat::kR8G8B8Unorm,
                          .max_width = static_cast<std::uint32_t>(options.model_width),
                          .max_height = static_cast<std::uint32_t>(options.model_height)});
  const auto holoviz = graph.op<viz::HolovizOp>(
      "holoviz",
      std::move(layers),
      viz::WindowSinkDesc{.title = "V4L2 -> TensorPreprocessor -> HoloInfer -> Holoviz",
                          .width = static_cast<std::uint32_t>(options.width),
                          .height = static_cast<std::uint32_t>(options.height)});

  const hs::ConnectionOptions mailbox{.queue_policy = hs::QueuePolicy::kDropOldest,
                                      .queue_depth = 1U};
  graph.add_flow(camera->frame, *preprocessor->image_input_port(), mailbox);
  graph.add_flow(*preprocessor->output_port(), *inference->input_port("model_input"), mailbox);
  graph.add_flow(*inference->output_port("prediction"), image->input, mailbox);
  graph.add_flow(image->frame, holoviz->image_layer_input(0), mailbox);
  graph.partition("display").add(holoviz);
  graph.auto_partition_remaining();
  graph.set_default_clock(graph.add_clock<hs::RealtimeClock>("clock"));

  // CUDA_VISIBLE_DEVICES selects the physical GPU. All stages use visible device 0.
  hs::CompileOptions compile_options;
  for (const auto& [op, port] :
       {std::pair{"camera", "frame"}, std::pair{"preprocessor", "model_input"},
        std::pair{"inference", "prediction"}}) {
    compile_options.deployment.bind_tensor_output_device(hs::TensorOutputDevicePlacement{
        .operator_path = op, .output_port = port, .device = hs::DeviceId{0}});
  }
  const hs::ExecutionPlan plan = hs::compile(graph, std::move(compile_options));
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (options.validate) {
    std::cout << "v4l2_inference_holoviz graph validation complete\n";
    return 0;
  }

  camera_inference::processed_frames.store(0);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds{options.timeout};
  auto session = hs::run_async(plan);
  while (session.is_running() && camera_inference::processed_frames.load() < options.frames &&
         std::chrono::steady_clock::now() < deadline) {
    static_cast<void>(session.wait_for(std::chrono::milliseconds{50}));
  }
  const auto observed = camera_inference::processed_frames.load();
  const bool timed_out = std::chrono::steady_clock::now() >= deadline;
  session.request_stop();
  session.wait();
  const bool passed = !timed_out && observed >= options.frames &&
                      session.termination() != hs::RunTermination::kFailed;
  std::cout << "inference frames: requested=" << options.frames << " observed=" << observed
            << " result=" << (passed ? "PASS" : "FAIL") << '\n';
  for (const auto& diagnostic : session.diagnostics()) {
    std::cerr << diagnostic.token << " @vertex=" << diagnostic.source.vertex.value << '\n';
  }
  return passed ? 0 : 2;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = camera_inference::parse_options(argc, argv);
    if (options.help) {
      std::cout << camera_inference::kUsage;
      return 0;
    }
    return run(options);
  } catch (const std::exception& error) {
    std::cerr << "v4l2_inference_holoviz: " << error.what() << '\n';
    return 1;
  }
}
