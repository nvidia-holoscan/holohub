// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/**
 * @file v4l2_depth.cu
 * @brief Live V4L2 to Depth Anything V2 pipeline for the Holoscan SDK 5.x API.
 *
 * The base video remains full-rate while the expensive depth branch processes
 * one frame out of every `--skip` inputs. CudaCompositorOp latches the newest
 * depth overlay and applies it to subsequent base frames:
 *
 *   V4L2Source -> YUYV-to-BGRA ---------------------------> Compositor -> X11
 *                         \-> Skipper -> Preprocess -> TensorRT -> Colorizer -/
 *
 * SDK 5.x has no implicit host/device bridge, so every payload is a native,
 * explicitly placed CUDA Tensor until X11DisplayOp performs the final D2H copy.
 */

#include <array>
#include <charconv>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/diagnostic.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "bgra_to_planar_tensor/bgra_to_planar_tensor.hpp"
#include "cuda_compositor/cuda_compositor.hpp"
#include "depth_colorizer/depth_colorizer.hpp"
#include "frame_skipper/frame_skipper.hpp"
#include "tensorrt_inference/tensorrt_inference.hpp"
#include "v4l2_source/v4l2_source.hpp"
#include "x11_display/x11_display.hpp"
#include "yuyv_to_bgra/yuyv_to_bgra.hpp"

namespace {

using holoscan::examples::v4l2_depth::CudaCompositorOp;
using holoscan::examples::v4l2_depth::DepthColorizerOp;
using holoscan::examples::v4l2_depth::FrameSkipperOp;
using holoscan::examples::v4l2_depth::BgraToPlanarTensorOp;
using holoscan::examples::v4l2_depth::TensorRtInferenceOp;
using holoscan::examples::v4l2_depth::V4L2SourceOp;
using holoscan::examples::v4l2_depth::X11DisplayOp;
using holoscan::examples::v4l2_depth::YuyvToBgraOp;

constexpr std::int32_t kNetworkSize = 518;
constexpr std::array<float, 3U> kImageNetMean{0.485F, 0.456F, 0.406F};
constexpr std::array<float, 3U> kImageNetStdDev{0.229F, 0.224F, 0.225F};
constexpr std::string_view kDefaultModel{"models/depth_anything_v2_vits.onnx"};
constexpr std::string_view kDefaultEngineCache =
    "depth_anything_v2_vits-715fade13be8-a5715a72-opset18-v1.engine";  // NOLINT

volatile std::sig_atomic_t g_stop_requested = 0;

struct Options {
  std::string device{"/dev/video0"};
  std::string data_directory{"data/v4l2_depth"};
  std::string model;
  std::string engine_cache;
  std::int32_t width{640};
  std::int32_t height{480};
  std::int32_t fps{30};
  std::uint32_t skip{2U};
  std::uint32_t duration_seconds{};
  std::int32_t cuda_device{0};
  bool validate_only{};
  bool show_help{};
};

void on_signal(int) { g_stop_requested = 1; }

void print_usage(std::ostream& output, std::string_view program) {
  output
      << "Usage: " << program << " [options]\n\n"
      << "Live V4L2 YUYV capture with a latched Depth Anything V2 overlay.\n\n"
      << "Options:\n"
      << "  --device PATH         V4L2 capture device (default: /dev/video0)\n"
      << "  --data-dir PATH       Model/cache root (default: data/v4l2_depth)\n"
      << "  --model PATH          Override the default ONNX model or use a TensorRT engine\n"
      << "  --engine-cache PATH   Override the default ONNX-built TensorRT engine cache\n"
      << "  --width PIXELS        Exact YUYV capture width (default: 640)\n"
      << "  --height PIXELS       Exact YUYV capture height (default: 480)\n"
      << "  --fps RATE            Exact capture rate (default: 30)\n"
      << "  --skip N              Infer on one frame out of N (default: 2)\n"
      << "  --duration SECONDS    Stop after a fixed duration (default: run until stopped)\n"
      << "  --cuda-device INDEX   CUDA device ordinal (default: 0)\n"
      << "  --validate            Compile the graph without opening camera/model/display\n"
      << "  -h, --help            Show this help text\n\n"
      << "Press q, Escape, or Ctrl-C to stop a live run.\n";
}

template <typename Integer>
[[nodiscard]] Integer parse_integer(std::string_view text, std::string_view option) {
  Integer value{};
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const auto [next, error] = std::from_chars(begin, end, value);
  if (error != std::errc() || next != end) {
    throw std::invalid_argument(std::string(option) + " requires an integer");
  }
  return value;
}

[[nodiscard]] std::string_view option_value(
    int& index, int argc, char** argv, std::string_view option) {
  if (index + 1 >= argc) {
    throw std::invalid_argument(std::string(option) + " requires a value");
  }
  return argv[++index];
}

[[nodiscard]] Options parse_options(int argc, char** argv) {
  Options options;
  bool model_overridden = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument{argv[index]};
    if (argument == "-h" || argument == "--help") {
      options.show_help = true;
    } else if (argument == "--validate") {
      options.validate_only = true;
    } else if (argument == "--device") {
      options.device = option_value(index, argc, argv, argument);
    } else if (argument == "--data-dir") {
      options.data_directory = option_value(index, argc, argv, argument);
    } else if (argument == "--model") {
      options.model = option_value(index, argc, argv, argument);
      model_overridden = true;
    } else if (argument == "--engine-cache") {
      options.engine_cache = option_value(index, argc, argv, argument);
    } else if (argument == "--width") {
      options.width =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--height") {
      options.height =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--fps") {
      options.fps =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--skip") {
      options.skip =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--duration") {
      options.duration_seconds =
          parse_integer<std::uint32_t>(option_value(index, argc, argv, argument), argument);
    } else if (argument == "--cuda-device") {
      options.cuda_device =
          parse_integer<std::int32_t>(option_value(index, argc, argv, argument), argument);
    } else {
      throw std::invalid_argument("unknown option: " + std::string(argument));
    }
  }

  if (options.device.empty() || options.data_directory.empty()) {
    throw std::invalid_argument("--device and --data-dir must not be empty");
  }
  if (options.model.empty()) {
    if (model_overridden) {
      throw std::invalid_argument("--model must not be empty");
    }
    if (options.validate_only) {
      // Graph compilation validates the model contract without opening this path.
      options.model = "v4l2_depth_validation_placeholder.onnx";
    } else {
      options.model =
          (std::filesystem::path(options.data_directory) / kDefaultModel).string();
      if (options.engine_cache.empty()) {
        options.engine_cache =
            (std::filesystem::path(options.data_directory) / kDefaultEngineCache).string();
      }
    }
  }
  if (options.width <= 0 || options.height <= 0 || options.fps <= 0 || options.skip == 0U ||
      options.cuda_device < 0) {
    throw std::invalid_argument(
        "--width, --height, --fps, and --skip must be positive; --cuda-device must be nonnegative");
  }
  if ((options.width & 1) != 0) {
    throw std::invalid_argument("--width must be even for packed YUYV");
  }
  return options;
}

void bind_tensor_output(holoscan::CompileOptions& compile_options,
                        std::string_view operator_path,
                        std::string_view output_port,
                        std::int32_t cuda_device) {
  compile_options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = operator_path,
      .output_port = output_port,
      .device = holoscan::DeviceId{cuda_device},
  });
}

[[nodiscard]] holoscan::ExecutionPlan build_plan(const Options& options) {
  holoscan::Graph graph{"v4l2-depth"};

  const auto camera =
      graph.op<V4L2SourceOp>("camera", options.device, options.width, options.height, options.fps);
  const auto convert =
      graph.op<YuyvToBgraOp>("convert", options.width, options.height);
  const auto skipper = graph.op<FrameSkipperOp>("skipper", options.skip);
  const auto preprocess = graph.op<BgraToPlanarTensorOp>("preprocess",
                                                        options.width,
                                                        options.height,
                                                        kNetworkSize,
                                                        kNetworkSize,
                                                        kImageNetMean,
                                                        kImageNetStdDev);
  const auto inference =
      graph.op<TensorRtInferenceOp>(
          "inference", options.model, options.engine_cache, options.cuda_device);
  const auto colorizer = graph.op<DepthColorizerOp>(
      "colorizer",
      options.width,
      options.height,
      kNetworkSize,
      kNetworkSize,
      150U,
      options.cuda_device);
  const auto compositor =
      graph.op<CudaCompositorOp>("compositor", options.width, options.height, options.cuda_device);
  const auto display =
      graph.op<X11DisplayOp>("display", options.width, options.height, "Holoscan V4L2 Depth");

  const holoscan::ConnectionOptions blocking_two{.queue_depth = 2U};
  const holoscan::ConnectionOptions latest_one{
      .queue_policy = holoscan::QueuePolicy::kDropOldest,
      .queue_depth = 1U,
  };

  graph.add_flow(camera->frame, convert->input, blocking_two);
  graph.add_flow(convert->output, compositor->base, latest_one);
  graph.add_flow(convert->output, skipper->input, latest_one);
  graph.add_flow(skipper->output, preprocess->input, latest_one);
  graph.add_flow(preprocess->output, inference->input, latest_one);
  graph.add_flow(inference->output, colorizer->input, latest_one);
  graph.add_flow(colorizer->output, compositor->overlay, latest_one);
  graph.add_flow(compositor->output, display->input, latest_one);
  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));

  // SDK 5.x deliberately does not infer CUDA device zero. Bind every output
  // that owns a plan-managed device allocation. The skipper only republishes
  // an existing Tensor loan, so it has no independent allocation binding.
  holoscan::CompileOptions compile_options;
  bind_tensor_output(compile_options, "camera", "frame", options.cuda_device);
  bind_tensor_output(compile_options, "convert", "output", options.cuda_device);
  bind_tensor_output(compile_options, "preprocess", "output", options.cuda_device);
  bind_tensor_output(compile_options, "inference", "output", options.cuda_device);
  bind_tensor_output(compile_options, "colorizer", "output", options.cuda_device);
  bind_tensor_output(compile_options, "compositor", "output", options.cuda_device);

  return holoscan::compile(graph, std::move(compile_options));
}

[[nodiscard]] int run(const Options& options) {
  holoscan::ExecutionPlan plan = build_plan(options);
  if (!plan.ok()) {
    std::cerr << plan.json() << '\n';
    return 1;
  }
  if (options.validate_only) {
    std::cout << "v4l2_depth graph validation complete\n";
    return 0;
  }

  g_stop_requested = 0;
  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);

  holoscan::RunSession session = holoscan::run_async(plan);
  const auto deadline =
      options.duration_seconds == 0U
          ? std::chrono::steady_clock::time_point::max()
          : std::chrono::steady_clock::now() +
                std::chrono::seconds{options.duration_seconds};
  while (session.is_running() && g_stop_requested == 0) {
    if (std::chrono::steady_clock::now() >= deadline) {
      g_stop_requested = 1;
      break;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds{20});
  }
  if (session.is_running()) {
    session.request_stop();
  }
  session.wait();

  const holoscan::RunTermination termination = session.termination();
  if (!session.diagnostics().empty()) {
    std::cerr << "v4l2_depth runtime diagnostics:\n";
    for (const holoscan::Diagnostic& diagnostic : session.diagnostics()) {
      std::cerr << "  " << diagnostic.token;
      if (!diagnostic.required.empty()) {
        std::cerr << " required=" << diagnostic.required;
      }
      if (!diagnostic.provided.empty()) {
        std::cerr << " provided=" << diagnostic.provided;
      }
      std::cerr << '\n';
    }
  }

  if (termination == holoscan::RunTermination::kFailed ||
      (!session.diagnostics().empty() && termination != holoscan::RunTermination::kStopped)) {
    return 2;
  }
  std::cout << "v4l2_depth stopped cleanly\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parse_options(argc, argv);
    if (options.show_help) {
      print_usage(std::cout, argc > 0 ? argv[0] : "v4l2_depth");
      return 0;
    }
    return run(options);
  } catch (const std::exception& exception) {
    std::cerr << "v4l2_depth: " << exception.what() << '\n';
    print_usage(std::cerr, argc > 0 ? argv[0] : "v4l2_depth");
    return 64;
  }
}
