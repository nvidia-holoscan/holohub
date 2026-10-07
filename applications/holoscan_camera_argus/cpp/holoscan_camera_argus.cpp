// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format off
#include <charconv>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/managed_shutdown_signals.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "argus_capture_op/argus_capture_op.hpp"
// clang-format on

namespace camera = holoscan::holoscan_camera;
namespace {
class FrameConsumer final : public holoscan::Operator<> {
 public:
  FrameConsumer(holoscan::MemoryKind memory, std::string save_path)
      : memory_(memory), save_path_(std::move(save_path)) {}
  holoscan::Input<holoscan::schema::ImageT> frame;
  void setup(holoscan::OperatorSpec& spec) override {
    spec.lifecycle().stage(holoscan::LifecycleStage::kStop, &FrameConsumer::on_stop);
    spec.input(frame, "frame")
        .queue_depth(2U)
        .expects_tensor(holoscan::TensorInputSpec{
            .representation = {
                .memory_kind = memory_, .dtype = DLDataType{kDLUInt, 8, 1}, .rank = 1U}});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnEach{frame});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto image = frame.receive();
    if (!image) return holoscan::make_unexpected(image.error());
    ++frames_;
    const auto& descriptor = image->data;
    if (!save_path_.empty() && !saved_frame_) {
      // The CLI requires host placement. Native capture completes the device-to-host copy
      // before publication, and the received sample keeps the pixels alive through this write.
      const auto* pixels = static_cast<const char*>(descriptor.data->data());
      if (!pixels) {
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "frame pixels are unavailable"});
      }
      std::ofstream output(save_path_, std::ios::binary | std::ios::trunc);
      output.write(pixels, static_cast<std::streamsize>(descriptor.data->nbytes()));
      output.close();
      if (!output) {
        std::cerr << "Argus sample: cannot write frame to " << save_path_ << '\n';
        return holoscan::make_unexpected(
            holoscan::Error{holoscan::ErrorCode::kFailure, "cannot write captured frame"});
      }
      saved_frame_ = true;
      std::cout << "Saved frame: " << save_path_ << ", " << descriptor.width << 'x'
                << descriptor.height << ' '
                << holoscan::schema::EnumNameImageEncoding(descriptor.encoding)
                << ", sequence=" << descriptor.header->device_sequence
                << ", bytes=" << descriptor.data->nbytes() << '\n';
    }
    // The descriptor and sample envelope are CPU-readable even when the pixels remain on GPU.
    if (frames_ == 1 || frames_ % 30 == 0) {
      std::cout << "frame " << frames_ << ": " << descriptor.width << 'x' << descriptor.height
                << ' ' << holoscan::schema::EnumNameImageEncoding(descriptor.encoding)
                << ", sequence=" << descriptor.header->device_sequence
                << ", bytes=" << descriptor.data->nbytes()
                << ", degraded=" << image->metadata.degraded() << '\n';
    }
    return {};
  }

  holoscan::LifecycleStatus on_stop(holoscan::LifecycleContext&) noexcept {
    if (!save_path_.empty() && !saved_frame_) {
      std::fprintf(stderr, "Argus sample: capture stopped without saving an image to %s\n",
                   save_path_.c_str());
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  }

 private:
  holoscan::MemoryKind memory_;
  std::string save_path_;
  bool saved_frame_{};
  std::uint64_t frames_{};
};

class MetadataConsumer final : public holoscan::Operator<> {
 public:
  explicit MetadataConsumer(std::uint64_t count) : count_(count) {}
  holoscan::Input<camera::ArgusFrameMetadataT> metadata;
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(metadata, "metadata").queue_depth(2U);
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnEach{metadata});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext& context) override {
    auto sample = metadata.receive();
    if (!sample) return holoscan::make_unexpected(sample.error());
    ++received_;
    if (received_ == 1 || received_ % 30 == 0) {
      const auto& data = sample->data;
      std::cout << "metadata: camera=" << data.camera_index << ", sequence=" << data.sequence
                << ", SOF(TSC ns)=" << data.sensor_sof_timestamp_ns
                << ", exposure(ns)=" << data.exposure_time_ns
                << ", queue drops=" << data.queue_drops
                << ", publication drops=" << data.publication_drops << '\n';
    }
    if (received_ >= count_) context.request_completion();
    return {};
  }

 private:
  std::uint64_t count_;
  std::uint64_t received_{};
};

template <class T>
T number(std::string_view value) {
  T result{};
  auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc() || parsed.ptr != value.data() + value.size()) {
    throw std::invalid_argument("invalid numeric option: " + std::string(value));
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  try {
    camera::ArgusCaptureConfig config;
    std::uint64_t count = 120;
    std::string save_path;
    bool validate = false;
    bool managed_signals = true;
    for (int i = 1; i < argc; ++i) {
      const std::string_view key = argv[i];
      if (key == "--validate") {
        validate = true;
        continue;
      }
      if (key == "--help") {
        std::cout << "holoscan_camera_argus [--validate] [--camera N] [--mode N] "
                     "[--width W] [--height H] [--fps F] [--format nv12|i420] "
                     "[--buffers N] [--drop oldest|newest] [--timeout-ms N] [--frames N] "
                     "[--memory device|host|pinned] [--integration-offset-ns N] "
                     "[--signal-policy managed|caller] [--save-frame FILE]\n"
                     "--save-frame writes the first received frame as raw NV12/I420, replacing "
                     "FILE; requires --memory host or pinned.\n";
        return 0;
      }
      if (++i == argc) throw std::invalid_argument("missing value for " + std::string(key));
      const std::string_view value = argv[i];
      if (value.empty()) throw std::invalid_argument("empty value for " + std::string(key));
      if (key == "--camera")
        config.camera_index = number<std::uint32_t>(value);
      else if (key == "--mode")
        config.sensor_mode = number<std::uint32_t>(value);
      else if (key == "--width")
        config.width = number<std::uint32_t>(value);
      else if (key == "--height")
        config.height = number<std::uint32_t>(value);
      else if (key == "--fps")
        config.fps = number<double>(value);
      else if (key == "--buffers")
        config.buffer_count = number<std::uint32_t>(value);
      else if (key == "--timeout-ms")
        config.timeout_ms = number<std::uint32_t>(value);
      else if (key == "--frames")
        count = number<std::uint64_t>(value);
      else if (key == "--save-frame")
        save_path = value;
      else if (key == "--integration-offset-ns")
        config.integration_start_offset_ns = number<std::int64_t>(value);
      else if (key == "--signal-policy" && value == "managed")
        managed_signals = true;
      else if (key == "--signal-policy" && value == "caller")
        managed_signals = false;
      else if (key == "--format" && value == "nv12")
        config.pixel_format = camera::ArgusPixelFormat::kNv12;
      else if (key == "--format" && value == "i420")
        config.pixel_format = camera::ArgusPixelFormat::kI420;
      else if (key == "--drop" && value == "oldest")
        config.drop_policy = camera::ArgusDropPolicy::kDropOldest;
      else if (key == "--drop" && value == "newest")
        config.drop_policy = camera::ArgusDropPolicy::kDropNewest;
      else if (key == "--memory" && value == "device")
        config.memory_kind = holoscan::MemoryKind::kCudaDevice;
      else if (key == "--memory" && value == "host")
        config.memory_kind = holoscan::MemoryKind::kHost;
      else if (key == "--memory" && value == "pinned")
        config.memory_kind = holoscan::MemoryKind::kPinnedHost;
      else
        throw std::invalid_argument("unknown option or value: " + std::string(key));
    }
    if (!count) throw std::invalid_argument("--frames must be positive");
    if (!save_path.empty() && config.memory_kind == holoscan::MemoryKind::kCudaDevice) {
      throw std::invalid_argument("--save-frame requires --memory host or --memory pinned");
    }
    // Runtime graph compilation can start CUDA threads. Establish their signal mask first.
    // Keep the domain alive until after the graph, plan, and capture session are retired.
    std::optional<holoscan::ManagedShutdownSignals> signals;
    if (!validate && managed_signals) {
      auto created = holoscan::ManagedShutdownSignals::create();
      if (!created) {
        std::cerr << "Argus signal setup: " << holoscan::runtime_error_message(created.error())
                  << '\n';
        if (created.error().code == holoscan::ErrorCode::kInvalidArgument) {
          std::cerr << "The process already has additional threads. Use --signal-policy caller "
                       "to leave SIGINT/SIGTERM handling to the process; Ctrl-C will not use "
                       "HSDK cooperative cleanup.\n";
        }
        return 1;
      }
      signals.emplace(std::move(created).value());
    }
    holoscan::Graph graph{"argus-camera"};
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("runtime-clock"));
    auto source = graph.op<camera::ArgusCaptureOp>("camera", config);
    auto frames = graph.op<FrameConsumer>("frames", config.memory_kind, save_path);
    auto metadata = graph.op<MetadataConsumer>("metadata", count);
    graph.add_flow(source->frame, frames->frame);
    graph.add_flow(source->metadata, metadata->metadata);
    holoscan::CompileOptions options;
    if (config.memory_kind == holoscan::MemoryKind::kCudaDevice) {
      options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
          .operator_path = "camera", .output_port = "frame", .device = holoscan::DeviceId{0}});
    }
    const auto plan = holoscan::compile(graph, std::move(options));
    if (!plan.ok()) {
      std::cerr << plan.json() << '\n';
      return 1;
    }
    if (validate) {
      std::cout << "Argus graph validation complete\n";
      return 0;
    }
    // Caller-managed execution supports processes whose libraries start threads before main().
    const auto policy = managed_signals ? holoscan::RunSignalPolicy::kManaged
                                        : holoscan::RunSignalPolicy::kCallerManaged;
    const auto status = holoscan::run(plan, policy);
    std::cout << "Argus capture finished (status=" << status << ")\n";
    return status;
  } catch (const std::exception& e) {
    std::cerr << "Argus sample: " << e.what() << '\n';
    return 1;
  }
}
