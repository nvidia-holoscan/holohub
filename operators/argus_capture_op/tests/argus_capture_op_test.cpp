// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format off
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <holoscan/core/compile.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/graph.hpp>
#include <holoscan/core/run.hpp>
#include <holoscan/time/realtime_clock.hpp>

#include "argus_capture_op/argus_capture_service.hpp"
// clang-format on

namespace camera = holoscan::holoscan_camera;
namespace {
struct State {
  std::mutex mutex;
  std::set<std::size_t> borrowed;
  std::vector<holoscan::schema::ImageT> images;
  std::vector<camera::ArgusFrameMetadataT> metadata;
  std::vector<bool> degraded;
  std::atomic<unsigned> opens{}, starts{}, stops{}, destroys{}, releases{};
  unsigned max_borrowed{};
  bool allocation_failure{}, start_failure{}, capture_failure{}, copy_failure{}, stall{};
  std::chrono::milliseconds consumer_delay{};
};
State* state;

std::int64_t now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

class FakeService final : public camera::argus::CaptureService {
 public:
  explicit FakeService(camera::ArgusCaptureConfig config) : config_(std::move(config)) {
    ++state->opens;
    if (state->allocation_failure) throw std::runtime_error("injected allocation failure");
  }
  ~FakeService() override { ++state->destroys; }
  void start() override {
    ++state->starts;
    if (state->start_failure) throw std::runtime_error("injected start failure");
    running_ = true;
  }
  void stop() override {
    if (running_) {
      ++state->stops;
      running_ = false;
    }
  }
  std::optional<camera::argus::CapturedFrame> acquire(std::uint32_t timeout_ms) override {
    if (state->capture_failure) throw std::runtime_error("injected stream failure");
    if (state->stall) {
      std::this_thread::sleep_for(std::chrono::milliseconds(timeout_ms));
      return std::nullopt;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
    std::lock_guard lock(state->mutex);
    for (std::size_t slot = 0; slot < config_.buffer_count; ++slot) {
      if (state->borrowed.contains(slot)) continue;
      state->borrowed.insert(slot);
      state->max_borrowed =
          std::max(state->max_borrowed, static_cast<unsigned>(state->borrowed.size()));
      camera::argus::CapturedFrame frame;
      frame.slot = slot;
      frame.metadata.sequence = ++sequence_;
      frame.metadata.camera_index = config_.camera_index;
      frame.metadata.sensor_mode = config_.sensor_mode;
      frame.metadata.width = config_.width;
      frame.metadata.height = config_.height;
      frame.metadata.sensor_sof_timestamp_ns = now_ns();
      frame.metadata.sensor_timestamp_ns = frame.metadata.sensor_sof_timestamp_ns;
      frame.metadata.timestamp_clock = camera::ArgusTimestampClock_TEGRA_TSC_NS;
      frame.metadata.exposure_time_ns = 1000;
      frame.metadata.analog_gain = 2;
      frame.metadata.isp_digital_gain = 1;
      return frame;
    }
    throw std::runtime_error("native pool starved: buffer ownership was not bounded");
  }
  void release(std::size_t slot) override {
    std::lock_guard lock(state->mutex);
    if (!state->borrowed.erase(slot)) throw std::runtime_error("double release");
    ++state->releases;
  }
  void copy_frame(std::size_t slot, const holoscan::TensorOutputWriteGuard& writer,
                  holoscan::MemoryKind placement) override {
    std::lock_guard lock(state->mutex);
    if (!state->borrowed.contains(slot)) throw std::runtime_error("copy after release");
    if (state->copy_failure) throw std::runtime_error("injected copy failure");
    if (placement != holoscan::MemoryKind::kHost)
      throw std::runtime_error("fake expects host storage");
    std::memset(writer.data(), 0x5a, camera::argus::validate_config(config_));
  }
  std::optional<std::int64_t> timestamp_now_ns() noexcept override { return now_ns(); }

 private:
  camera::ArgusCaptureConfig config_;
  std::uint64_t sequence_{};
  bool running_{};
};

class ImageSink final : public holoscan::Operator<> {
 public:
  holoscan::Input<holoscan::schema::ImageT> input;
  void setup(holoscan::OperatorSpec& spec) override {
    spec.input(input, "input")
        .queue_depth(2U)
        .expects_tensor(
            holoscan::TensorInputSpec{.representation = {.memory_kind = holoscan::MemoryKind::kHost,
                                                         .dtype = DLDataType{kDLUInt, 8, 1},
                                                         .rank = 1U}});
  }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnEach{input});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) return holoscan::make_unexpected(sample.error());
    if (state->consumer_delay.count()) std::this_thread::sleep_for(state->consumer_delay);
    const auto& image = sample->data;
    EXPECT_NE(image.data, nullptr);
    if (image.data) {
      EXPECT_EQ(image.data->nbytes(), 48U);
      EXPECT_EQ(*static_cast<const std::uint8_t*>(image.data->data()), 0x5a);
    }
    std::lock_guard lock(state->mutex);
    // Keep the descriptor, never a tensor loan: holding every tensor would exhaust the pool.
    auto descriptor = image;
    descriptor.data.reset();
    state->images.push_back(std::move(descriptor));
    state->degraded.push_back(sample->metadata.degraded());
    return {};
  }
};

class MetadataSink final : public holoscan::Operator<> {
 public:
  holoscan::Input<camera::ArgusFrameMetadataT> input;
  void setup(holoscan::OperatorSpec& spec) override { spec.input(input, "input").queue_depth(2U); }
  holoscan::Contract contract() const override {
    holoscan::Contract c;
    c.trigger(holoscan::OnEach{input});
    return c;
  }
  holoscan::expected<void, holoscan::Error> compute(holoscan::ExecutionContext&) override {
    auto sample = input.receive();
    if (!sample) return holoscan::make_unexpected(sample.error());
    std::lock_guard lock(state->mutex);
    state->metadata.push_back(sample->data);
    return {};
  }
};

class ArgusCaptureTest : public ::testing::Test {
 protected:
  State owned_;
  camera::ArgusCaptureConfig config_;
  void SetUp() override {
    state = &owned_;
    config_.width = 8;
    config_.height = 4;
    config_.camera_index = 2;
    config_.sensor_mode = 3;
    config_.memory_kind = holoscan::MemoryKind::kHost;
  }
  holoscan::RunTermination run(std::size_t wanted = 6) {
    holoscan::Graph graph{"argus-test"};
    graph.set_default_clock(graph.add_clock<holoscan::RealtimeClock>("clock"));
    auto source = graph.op<camera::ArgusCaptureOp>("camera", config_);
    auto images = graph.op<ImageSink>("images");
    auto metadata = graph.op<MetadataSink>("metadata");
    graph.add_flow(source->frame, images->input);
    graph.add_flow(source->metadata, metadata->input);
    auto plan = holoscan::compile(graph);
    EXPECT_TRUE(plan.ok()) << plan.json();
    EXPECT_EQ(owned_.opens.load(), 0U) << "graph authoring must not open the camera";
    auto session = holoscan::run_async(plan);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    bool received = false;
    while (session.is_running() && std::chrono::steady_clock::now() < deadline) {
      {
        std::lock_guard lock(owned_.mutex);
        received = owned_.images.size() >= wanted && owned_.metadata.size() >= wanted;
      }
      if (received) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    session.request_stop();
    session.wait();
    EXPECT_TRUE(owned_.borrowed.empty()) << "teardown must return every native buffer";
    EXPECT_LE(owned_.max_borrowed, config_.buffer_count);
    if (session.termination() == holoscan::RunTermination::kFailed) {
      EXPECT_FALSE(session.diagnostics().empty());
    } else {
      EXPECT_TRUE(received) << "operator did not deliver frames before the deadline";
    }
    return session.termination();
  }
};

TEST_F(ArgusCaptureTest, PublishesNv12DescriptorsAndCorrelatedMetadata) {
  EXPECT_NE(run(), holoscan::RunTermination::kFailed);
  EXPECT_EQ(owned_.starts, 1U);
  EXPECT_EQ(owned_.stops, 1U);
  EXPECT_EQ(owned_.destroys, 1U);
  std::size_t correlated = 0;
  for (const auto& image : owned_.images) {
    EXPECT_EQ(image.encoding, holoscan::schema::ImageEncoding_NV12);
    EXPECT_EQ(image.width, 8);
    EXPECT_EQ(image.height, 4);
    ASSERT_EQ(image.plane_layouts.size(), 2U);
    EXPECT_EQ(image.plane_layouts[1].offset_bytes(), 32U);
    EXPECT_EQ(image.header->capture_timestamp_ns, 0);
    auto metadata =
        std::find_if(owned_.metadata.begin(), owned_.metadata.end(),
                     [&](const auto& m) { return m.sequence == image.header->device_sequence; });
    if (metadata != owned_.metadata.end()) {
      ++correlated;
      EXPECT_EQ(metadata->camera_index, 2U);
      EXPECT_EQ(metadata->sensor_mode, 3U);
      EXPECT_GT(metadata->sensor_timestamp_ns, 0U);
      EXPECT_FALSE(metadata->capture_time_valid);
    }
  }
  EXPECT_GT(correlated, 0U) << "image/metadata sequences never matched";
}

TEST_F(ArgusCaptureTest, PublishesI420WithExplicitPlaneOffsets) {
  config_.pixel_format = camera::ArgusPixelFormat::kI420;
  EXPECT_NE(run(), holoscan::RunTermination::kFailed);
  ASSERT_FALSE(owned_.images.empty());
  const auto& image = owned_.images.front();
  EXPECT_EQ(image.encoding, holoscan::schema::ImageEncoding_I420);
  ASSERT_EQ(image.plane_layouts.size(), 3U);
  EXPECT_EQ(image.plane_layouts[1].offset_bytes(), 32U);
  EXPECT_EQ(image.plane_layouts[2].offset_bytes(), 40U);
}

TEST_F(ArgusCaptureTest, ProjectsOnlyCalibratedIntegrationTimestamps) {
  config_.integration_start_offset_ns = -1000;
  EXPECT_NE(run(), holoscan::RunTermination::kFailed);
  ASSERT_FALSE(owned_.metadata.empty());
  std::size_t correlated = 0;
  for (const auto& metadata : owned_.metadata) {
    EXPECT_TRUE(metadata.capture_time_valid);
    EXPECT_GT(metadata.capture_timestamp_ns, 0);
    for (const auto& image : owned_.images) {
      if (image.header->device_sequence == metadata.sequence) {
        ++correlated;
        EXPECT_EQ(image.header->capture_timestamp_ns, metadata.capture_timestamp_ns);
      }
    }
  }
  EXPECT_GT(correlated, 0U) << "calibrated image/metadata sequences never matched";
}

TEST_F(ArgusCaptureTest, SlowConsumerDropsFramesWithBoundedNativeOwnership) {
  config_.buffer_count = 4;
  owned_.consumer_delay = std::chrono::milliseconds(20);
  EXPECT_NE(run(), holoscan::RunTermination::kFailed);
  ASSERT_FALSE(owned_.metadata.empty());
  EXPECT_GT(owned_.metadata.back().queue_drops + owned_.metadata.back().publication_drops, 0U);
}

TEST_F(ArgusCaptureTest, AllocationFailureFailsTheSession) {
  owned_.allocation_failure = true;
  EXPECT_EQ(run(), holoscan::RunTermination::kFailed);
  EXPECT_EQ(owned_.starts, 0U);
}
TEST_F(ArgusCaptureTest, StartFailureReleasesTheSession) {
  owned_.start_failure = true;
  EXPECT_EQ(run(), holoscan::RunTermination::kFailed);
  EXPECT_EQ(owned_.destroys, 1U);
}
TEST_F(ArgusCaptureTest, StreamFailureWakesAndFailsTheGraph) {
  owned_.capture_failure = true;
  EXPECT_EQ(run(), holoscan::RunTermination::kFailed);
  EXPECT_EQ(owned_.stops, 1U);
}
TEST_F(ArgusCaptureTest, CopyFailureReturnsTheBorrowedBuffer) {
  owned_.copy_failure = true;
  EXPECT_EQ(run(), holoscan::RunTermination::kFailed);
  EXPECT_GT(owned_.releases, 0U);
}
TEST_F(ArgusCaptureTest, PersistentTimeoutFailsInsteadOfIdlingForever) {
  owned_.stall = true;
  config_.timeout_ms = 100;
  EXPECT_EQ(run(), holoscan::RunTermination::kFailed);
}

TEST(ArgusConfig, RejectsUnsupportedAndUnboundedConfigurations) {
  camera::ArgusCaptureConfig c;
  c.width = 7;
  EXPECT_THROW(camera::ArgusCaptureOp{c}, std::invalid_argument);
  c = {};
  c.buffer_count = 2;
  EXPECT_THROW(camera::ArgusCaptureOp{c}, std::invalid_argument);
  c = {};
  c.buffer_count = 65;
  EXPECT_THROW(camera::ArgusCaptureOp{c}, std::invalid_argument);
  c = {};
  c.fps = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(camera::ArgusCaptureOp{c}, std::invalid_argument);
  c = {};
  c.pixel_format = static_cast<camera::ArgusPixelFormat>(9);
  EXPECT_THROW(camera::ArgusCaptureOp{c}, std::invalid_argument);
  c = {};
  c.timeout_ms = 0;
  EXPECT_THROW(camera::ArgusCaptureOp{c}, std::invalid_argument);
}

TEST(ArgusQueue, DropOldestReturnsExactlyTheDisplacedNativeBuffer) {
  camera::argus::FrameQueue queue(2);
  EXPECT_FALSE(queue.push({.slot = 1}, camera::ArgusDropPolicy::kDropOldest));
  EXPECT_FALSE(queue.push({.slot = 2}, camera::ArgusDropPolicy::kDropOldest));
  auto dropped = queue.push({.slot = 3}, camera::ArgusDropPolicy::kDropOldest);
  ASSERT_TRUE(dropped);
  EXPECT_EQ(dropped->slot, 1U);
  EXPECT_EQ(queue.size(), 2U);
  EXPECT_EQ(queue.pop()->slot, 2U);
  EXPECT_EQ(queue.pop()->slot, 3U);
  EXPECT_FALSE(queue.pop());
}
TEST(ArgusQueue, DropNewestPreservesTheQueuedFrames) {
  camera::argus::FrameQueue queue(2);
  queue.push({.slot = 1}, camera::ArgusDropPolicy::kDropNewest);
  queue.push({.slot = 2}, camera::ArgusDropPolicy::kDropNewest);
  EXPECT_EQ(queue.push({.slot = 3}, camera::ArgusDropPolicy::kDropNewest)->slot, 3U);
  EXPECT_EQ(queue.pop()->slot, 1U);
  EXPECT_EQ(queue.pop()->slot, 2U);
}
}  // namespace

namespace holoscan::holoscan_camera::argus {
std::unique_ptr<CaptureService> make_service(const ArgusCaptureConfig& config) {
  return std::make_unique<FakeService>(config);
}
}  // namespace holoscan::holoscan_camera::argus
