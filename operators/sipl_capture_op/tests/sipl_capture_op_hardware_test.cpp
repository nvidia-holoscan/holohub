// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Integration tests for SIPLCaptureOp requiring a connected SIPL camera.
//
// All tests skip automatically when the required environment variables are not set:
//
//   SIPL_CAMERA_CONFIG   SIPL camera configuration name (e.g. "ov2311_raw", "imx274_raw").
//                        Must match a config known to the SIPL camera database, or a name
//                        defined inside the JSON file given by SIPL_JSON_CONFIG.
//   SIPL_JSON_CONFIG     Optional path to a vendor JSON platform config file.  When set the
//                        camera database is extended with the configs defined in that file.
//   SIPL_CAMERA_INDEX    Zero-based camera index within the rig (default: 0).
//   SIPL_CUDA_DEVICE     CUDA device index for NvSci buffer import (default: 0).
//
// Run with hardware and a built-in config:
//   SIPL_CAMERA_CONFIG=ov2311_raw ctest -L hardware
//
// Run with a vendor JSON platform config:
//   SIPL_CAMERA_CONFIG=<name_in_json> SIPL_JSON_CONFIG=/path/to/config.json ctest -L hardware

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/connection_options.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/core/run_session.hpp>
#include <holoscan/core/sample.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include <sipl_capture_op/sipl_capture_op.hpp>
#include <sipl_capture_op/sipl_capture_service.hpp>

namespace mm = holoscan::holoscan_camera;

// ---------------------------------------------------------------------------
// Shared results — written by sink operators, read by the test.
// ---------------------------------------------------------------------------

struct SIPLTestResults {
  std::atomic<int> frames{ 0 };
  std::atomic<int> sensor_data_records{ 0 };

  std::atomic<bool> done{ false };
  std::mutex mutex;
  std::condition_variable cv;
  int expect_frames{ 5 };

  // Called by both sink ops after incrementing their own counter. frame and sensor_data are
  // emitted together per-frame but travel to these two independently-scheduled consumers over
  // separate queues, so one can lag the other; only declaring done once both have reached
  // expect_frames avoids a spurious failure in ReceivesFramesOnBothPorts's lock-step assertion
  // from tearing the session down while sensor_data_sink still has a frame left to process.
  void maybe_finish() {
    if (frames.load() >= expect_frames && sensor_data_records.load() >= expect_frames) {
      done = true;
      cv.notify_all();
    }
  }

  bool wait(std::chrono::seconds timeout) {
    std::unique_lock<std::mutex> lock(mutex);
    return cv.wait_for(lock, timeout, [this] { return done.load(); });
  }
};

// ---------------------------------------------------------------------------
// Sink operators — one per SIPLCaptureOp output port.
// ---------------------------------------------------------------------------

struct FrameSinkParams {
  std::shared_ptr<SIPLTestResults> results;
};

class FrameSinkOp final : public holoscan::Operator<FrameSinkParams> {
 public:
  holoscan::Input<holoscan::schema::ImageT> input;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnEach{ input });
    return c;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    auto& r = *params().results;
    r.frames++;
    r.maybe_finish();
    return {};
  }
};

struct SensorDataSinkParams {
  std::shared_ptr<SIPLTestResults> results;
};

class SensorDataSinkOp final : public holoscan::Operator<SensorDataSinkParams> {
 public:
  holoscan::Input<mm::SIPLFrameMetadataT> input;

  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input").queue_depth(4U);
  }

  [[nodiscard]] holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnEach{ input });
    return c;
  }

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) {
      return holoscan::make_unexpected(std::move(sample).error());
    }
    auto& r = *params().results;
    r.sensor_data_records++;
    r.maybe_finish();
    return {};
  }
};

// ---------------------------------------------------------------------------
// Test fixture
// ---------------------------------------------------------------------------

class SIPLCaptureOpHardwareTest : public ::testing::Test {
 protected:
  static constexpr int kFramesToVerify = 5;
  static constexpr std::chrono::seconds kTimeout{ 30 };

  std::string camera_config;
  std::string json_config;
  std::uint32_t camera_index{ 0 };
  std::int32_t cuda_device{ 0 };

  void SetUp() override {
    const char* cfg = std::getenv("SIPL_CAMERA_CONFIG");
    if (!cfg || cfg[0] == '\0') {
      GTEST_SKIP() << "Set SIPL_CAMERA_CONFIG to run hardware tests "
                   << "(e.g. SIPL_CAMERA_CONFIG=ov2311_raw)";
    }
    camera_config = cfg;

    if (const char* json = std::getenv("SIPL_JSON_CONFIG")) {
      json_config = json;
    }
    if (const char* idx = std::getenv("SIPL_CAMERA_INDEX")) {
      camera_index = static_cast<std::uint32_t>(std::atoi(idx));
    }
    if (const char* dev = std::getenv("SIPL_CUDA_DEVICE")) {
      cuda_device = static_cast<std::int32_t>(std::atoi(dev));
    }
  }

  [[nodiscard]] std::shared_ptr<mm::SIPLCaptureService> make_service() const {
    return std::make_shared<mm::SIPLCaptureService>(
        camera_config,
        json_config,
        /*raw_output=*/true,
        /*capture_queue_depth=*/4U,
        /*nito_base_path=*/"/var/nvidia/nvcam/settings/sipl",
        /*timeout_us=*/1'000'000U,
        cuda_device);
  }
};

// ---------------------------------------------------------------------------
// Graph validation — both ports wired, no streaming.
// ---------------------------------------------------------------------------

TEST_F(SIPLCaptureOpHardwareTest, GraphCompilationSucceedsWithAllPorts) {
  auto service = make_service();

  holoscan::Graph graph{ "sipl_all_ports_validation" };

  auto camera = graph.op<mm::SIPLCaptureOp>("camera", service, camera_index);

  // nullptr results are safe here: compile() validates topology only, compute() never runs.
  auto frame_sink = graph.op<FrameSinkOp>("frame_sink");
  frame_sink.set_params(FrameSinkParams{ .results = nullptr });

  auto sensor_data_sink = graph.op<SensorDataSinkOp>("sensor_data_sink");
  sensor_data_sink.set_params(SensorDataSinkParams{ .results = nullptr });

  graph.add_flow(camera->frame, frame_sink->input);
  graph.add_flow(camera->sensor_data, sensor_data_sink->input,
      holoscan::ConnectionOptions{ .queue_depth = 4U });

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions compile_opts;
  compile_opts.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "camera",
      .output_port = "frame",
      .device = holoscan::DeviceId{.value = cuda_device},
  });
  const auto plan = holoscan::compile(graph, std::move(compile_opts));
  EXPECT_TRUE(plan.ok()) << "Graph compilation failed:\n" << plan.json();
}

// ---------------------------------------------------------------------------
// Graph validation — kHost placement, no device binding.
// ---------------------------------------------------------------------------
//
// The device binding is REQUIRED for kCudaDevice and REFUSED for the two host kinds (see
// SIPLCaptureOp's memory_kind constructor parameter); this is the negative half of that pairing,
// exercised by simply not calling bind_tensor_output_device at all.

TEST_F(SIPLCaptureOpHardwareTest, GraphCompilationSucceedsWithHostPlacement) {
  auto service = make_service();

  holoscan::Graph graph{ "sipl_host_placement_validation" };

  auto camera = graph.op<mm::SIPLCaptureOp>(
      "camera", service, camera_index, holoscan::MemoryKind::kHost);

  auto frame_sink = graph.op<FrameSinkOp>("frame_sink");
  frame_sink.set_params(FrameSinkParams{ .results = nullptr });

  auto sensor_data_sink = graph.op<SensorDataSinkOp>("sensor_data_sink");
  sensor_data_sink.set_params(SensorDataSinkParams{ .results = nullptr });

  graph.add_flow(camera->frame, frame_sink->input);
  graph.add_flow(camera->sensor_data, sensor_data_sink->input,
      holoscan::ConnectionOptions{ .queue_depth = 4U });

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  const auto plan = holoscan::compile(graph, holoscan::CompileOptions{});
  EXPECT_TRUE(plan.ok()) << "Graph compilation failed:\n" << plan.json();
}

// ---------------------------------------------------------------------------
// Live streaming — frames arrive on both ports.
// ---------------------------------------------------------------------------

TEST_F(SIPLCaptureOpHardwareTest, ReceivesFramesOnBothPorts) {
  auto service = make_service();
  auto results = std::make_shared<SIPLTestResults>();
  results->expect_frames = kFramesToVerify;

  holoscan::Graph graph{ "sipl_streaming_test" };

  auto camera = graph.op<mm::SIPLCaptureOp>("camera", service, camera_index);

  auto frame_sink = graph.op<FrameSinkOp>("frame_sink");
  frame_sink.set_params(FrameSinkParams{ .results = results });

  auto sensor_data_sink = graph.op<SensorDataSinkOp>("sensor_data_sink");
  sensor_data_sink.set_params(SensorDataSinkParams{ .results = results });

  graph.add_flow(camera->frame, frame_sink->input);
  graph.add_flow(camera->sensor_data, sensor_data_sink->input,
      holoscan::ConnectionOptions{ .queue_depth = 4U });

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  holoscan::CompileOptions compile_opts;
  compile_opts.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "camera",
      .output_port = "frame",
      .device = holoscan::DeviceId{.value = cuda_device},
  });
  const auto plan = holoscan::compile(graph, std::move(compile_opts));
  ASSERT_TRUE(plan.ok()) << "Graph compilation failed:\n" << plan.json();

  auto session = holoscan::run_async(plan);
  const bool got_frames = results->wait(kTimeout);
  session.request_stop();
  session.wait();

  EXPECT_TRUE(got_frames)
      << "Timed out after " << kTimeout.count() << "s. "
      << "Received " << results->frames.load() << "/" << kFramesToVerify << " frames.";

  // frame and sensor_data are emitted in lock-step every compute().
  EXPECT_EQ(results->sensor_data_records.load(), results->frames.load())
      << "sensor_data port must emit once per frame";
}

// ---------------------------------------------------------------------------
// on_discover rejects out-of-range camera_index before streaming begins.
// ---------------------------------------------------------------------------

TEST_F(SIPLCaptureOpHardwareTest, OnDiscoverRejectsOutOfRangeCameraIndex) {
  auto service = make_service();

  // Force an obviously invalid index — well beyond any realistic rig size.
  constexpr std::uint32_t kBadIndex = 999U;

  holoscan::Graph graph{ "sipl_discover_reject_test" };

  auto camera = graph.op<mm::SIPLCaptureOp>("camera", service, kBadIndex);

  // Wire a minimal sink so the graph compiles.
  auto frame_sink = graph.op<FrameSinkOp>("frame_sink");
  frame_sink.set_params(FrameSinkParams{ .results = nullptr });
  graph.add_flow(camera->frame, frame_sink->input);

  graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));

  // Compilation validates topology only; on_discover is a runtime lifecycle stage.
  holoscan::CompileOptions compile_opts;
  compile_opts.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
      .operator_path = "camera",
      .output_port = "frame",
      .device = holoscan::DeviceId{.value = cuda_device},
  });
  const auto plan = holoscan::compile(graph, std::move(compile_opts));
  ASSERT_TRUE(plan.ok()) << "Graph compilation should succeed regardless of camera_index";

  // on_discover runs before streaming starts. When it returns kFatalFailure the runtime
  // stops the graph on its own. Poll until it stops or the deadline is reached.
  auto session = holoscan::run_async(plan);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(20);
  while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }

  const bool still_running = session.is_running();
  session.request_stop();
  session.wait();

  // A valid camera_index would stream indefinitely; a bad one must stop by itself.
  EXPECT_FALSE(still_running)
      << "Session should have stopped after on_discover rejected camera_index " << kBadIndex;
}
