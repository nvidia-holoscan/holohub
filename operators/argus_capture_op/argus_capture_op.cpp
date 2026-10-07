// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
// clang-format off
#include "argus_capture_op/argus_capture_op.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <utility>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/sensor_io.hpp>

#include "argus_capture_op/argus_capture_service.hpp"
// clang-format on

namespace holoscan::holoscan_camera {
namespace {
constexpr DLDataType kByte{kDLUInt, 8, 1};

// Worker polling policy: 100 ms trades stop/notification responsiveness for roughly ten
// timeout wakeups per second on an idle stream. This is independent of the sensor frame rate;
// acquire() returns early when a frame arrives. The requested wait assumes the backend honors
// its timeout; config.timeout_ms separately controls the continuous no-frame failure deadline.
constexpr std::uint32_t kAcquirePollTimeoutMs = 100;

// Failure-notification retry policy: a 1 ms backoff avoids busy-spinning at the start barrier
// or under backpressure while retrying promptly. This is a scheduler choice, not sensor timing.
constexpr auto kNotificationRetryDelay = std::chrono::milliseconds(1);

holoscan::LifecycleStatus stage_failure(const char* stage, const char* message) noexcept {
  std::fprintf(stderr, "ArgusCaptureOp %s: %s\n", stage, message);
  return holoscan::LifecycleStatus::kFatalFailure;
}

bool dropped_publication(holoscan::ErrorCode code) {
  return code == holoscan::ErrorCode::kBackpressured ||
         code == holoscan::ErrorCode::kResourceExhausted ||
         code == holoscan::ErrorCode::kPublicationIndeterminate;
}
}  // namespace

std::size_t argus::validate_config(const ArgusCaptureConfig& c) {
  if (c.width == 0 || c.height == 0 || (c.width & 1U) || (c.height & 1U) || c.width > 16384 ||
      c.height > 16384) {
    throw std::invalid_argument("Argus width and height must be even and in [2, 16384]");
  }
  if (!std::isfinite(c.fps) || c.fps < 0.1 || c.fps > 1000) {
    throw std::invalid_argument("Argus fps must be finite and in [0.1, 1000]");
  }
  if (c.buffer_count < 3 || c.buffer_count > 64) {
    throw std::invalid_argument("Argus buffer_count must be in [3, 64]");
  }
  // The no-frame deadline must allow at least one acquire poll and one expected frame period.
  if (c.timeout_ms < kAcquirePollTimeoutMs || c.timeout_ms > 60000 ||
      c.timeout_ms < 1000.0 / c.fps) {
    throw std::invalid_argument(
        "Argus timeout_ms must cover a frame period and be in [100, 60000]");
  }
  if (c.cuda_device != 0) {
    throw std::invalid_argument("Argus capture currently supports the Jetson integrated GPU (0)");
  }
  if (c.frame_id.empty() || c.frame_id.size() > 256) {
    throw std::invalid_argument("Argus frame_id must contain 1 to 256 bytes");
  }
  if (c.pixel_format != ArgusPixelFormat::kNv12 && c.pixel_format != ArgusPixelFormat::kI420) {
    throw std::invalid_argument("Argus pixel_format must be NV12 or I420");
  }
  if (c.drop_policy != ArgusDropPolicy::kDropOldest &&
      c.drop_policy != ArgusDropPolicy::kDropNewest) {
    throw std::invalid_argument("Argus drop_policy must be drop-oldest or drop-newest");
  }
  if (c.memory_kind != holoscan::MemoryKind::kCudaDevice &&
      c.memory_kind != holoscan::MemoryKind::kHost &&
      c.memory_kind != holoscan::MemoryKind::kPinnedHost) {
    throw std::invalid_argument("Argus memory_kind must be device, host, or pinned host");
  }
  return static_cast<std::size_t>(c.width) * c.height * 3 / 2;
}

struct ArgusCaptureOp::Impl {
  explicit Impl(const ArgusCaptureConfig& c) : config(c), queue(c.buffer_count - 2) {}
  ~Impl() {
    try {
      stop();
    } catch (const std::exception& e) {
      stage_failure("destructor", e.what());
    }
  }

  ArgusCaptureConfig config;
  std::unique_ptr<argus::CaptureService> service;
  argus::FrameQueue queue;
  std::mutex mutex;
  holoscan::NotificationSender sender;
  std::jthread worker;
  bool notification_pending{};
  std::exception_ptr failure;
  std::atomic<holoscan::ErrorCode> failure_code{holoscan::ErrorCode::kFailure};
  std::atomic<std::uint64_t> queue_drops{};
  std::atomic<std::uint64_t> timeouts{};
  std::uint64_t publication_drops{};
  holoscan::sensor_io::StageGuards guards;
  holoscan::sensor_io::ClockDiscipline clock;
  holoscan::sensor_io::SequenceTracker sequence;

  void notify() {
    {
      std::lock_guard lock(mutex);
      if (notification_pending || (queue.empty() && !failure)) return;
      notification_pending = true;
    }
    auto posted = sender.post();
    if (!posted) {
      std::lock_guard lock(mutex);
      notification_pending = false;
      const auto code = posted.error().code;
      if (code == holoscan::ErrorCode::kStaleEndpoint) {
        throw std::runtime_error(
            "stale readiness sender: restart ArgusCaptureOp at kArm or earlier");
      }
      if (code != holoscan::ErrorCode::kNotReady && code != holoscan::ErrorCode::kBackpressured &&
          code != holoscan::ErrorCode::kClosed) {
        throw std::runtime_error("failed to post Argus readiness notification");
      }
    }
  }

  void read(const std::stop_token& stop) noexcept {
    try {
      auto last_frame = std::chrono::steady_clock::now();
      while (!stop.stop_requested()) {
        notify();
        auto next = service->acquire(kAcquirePollTimeoutMs);
        if (!next) {
          ++timeouts;
          if (std::chrono::steady_clock::now() - last_frame >=
              std::chrono::milliseconds(config.timeout_ms)) {
            failure_code = holoscan::ErrorCode::kTimeout;
            throw std::runtime_error("camera produced no frame within timeout_ms");
          }
          continue;
        }
        last_frame = std::chrono::steady_clock::now();
        if (stop.stop_requested()) {
          service->release(next->slot);
          break;
        }
        std::optional<argus::CapturedFrame> dropped;
        {
          std::lock_guard lock(mutex);
          dropped = queue.push(std::move(*next), config.drop_policy);
        }
        if (dropped) {
          ++queue_drops;
          service->release(dropped->slot);
        }
        notify();
      }
    } catch (...) {
      {
        std::lock_guard lock(mutex);
        failure = std::current_exception();
      }
      // A failure needs an activation even if capture has stopped. Retry during the start
      // barrier/backpressure; never turn a device failure into a source that silently idles.
      while (!stop.stop_requested()) {
        try {
          notify();
        } catch (const std::exception& e) {
          stage_failure("readiness", e.what());
          break;
        }
        std::this_thread::sleep_for(kNotificationRetryDelay);
      }
    }
  }

  void stop() {
    if (worker.joinable()) {
      worker.request_stop();
      worker.join();
    }
    // Core quiesces compute before kStop, so no activation still owns a native buffer.
    std::exception_ptr first_error;
    while (auto held = queue.pop()) {
      try {
        service->release(held->slot);
      } catch (...) {
        if (!first_error) first_error = std::current_exception();
      }
    }
    if (service) {
      try {
        service->stop();
      } catch (...) {
        if (!first_error) first_error = std::current_exception();
      }
    }
    notification_pending = false;
    if (first_error) std::rethrow_exception(first_error);
  }
};

ArgusCaptureOp::ArgusCaptureOp(ArgusCaptureConfig config)
    : config_(std::move(config)),
      frame_bytes_(argus::validate_config(config_)),
      impl_(std::make_unique<Impl>(config_)) {
  // validate_config() runs before allocating the bounded capture queue.
}
ArgusCaptureOp::~ArgusCaptureOp() = default;

void ArgusCaptureOp::setup(holoscan::OperatorSpec& spec) {
  spec.output(frame, "frame")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {.memory_kind = config_.memory_kind, .dtype = kByte, .rank = 1U},
          .bounds = holoscan::tensor_bounds(frame_bytes_),
          .storage = holoscan::TensorOutputStorage::kRuntimePool});
  spec.output(metadata, "metadata").max_emits_per_compute(1U);
  spec.notification_source(frame_ready_, "argus-frame-ready")
      .capacity(1U)
      .sender_reference_capacity(2U);
  spec.lifecycle()
      .stage(holoscan::LifecycleStage::kConfigure, &ArgusCaptureOp::on_configure)
      .stage(holoscan::LifecycleStage::kAllocate, &ArgusCaptureOp::on_allocate)
      .stage(holoscan::LifecycleStage::kArm, &ArgusCaptureOp::on_arm)
      .stage(holoscan::LifecycleStage::kStart, &ArgusCaptureOp::on_start)
      .stage(holoscan::LifecycleStage::kStop, &ArgusCaptureOp::on_stop)
      .stage(holoscan::LifecycleStage::kRelease, &ArgusCaptureOp::on_release);
}

holoscan::Contract ArgusCaptureOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnNotified{.event = frame_ready_});
  return result;
}

holoscan::LifecycleStatus ArgusCaptureOp::on_configure(holoscan::LifecycleContext&) noexcept {
  return impl_->guards.run<holoscan::LifecycleStage::kConfigure>([this]() noexcept {
    impl_->clock.unbind();
    return holoscan::LifecycleStatus::kOk;
  });
}

holoscan::LifecycleStatus ArgusCaptureOp::on_allocate(holoscan::LifecycleContext&) noexcept {
  return impl_->guards.run<holoscan::LifecycleStage::kAllocate>([this]() noexcept {
    try {
      impl_->service = argus::make_service(config_);
      return holoscan::LifecycleStatus::kOk;
    } catch (const std::exception& e) {
      return stage_failure("allocate", e.what());
    } catch (...) {
      return stage_failure("allocate", "unknown backend failure");
    }
  });
}

holoscan::LifecycleStatus ArgusCaptureOp::on_arm(holoscan::LifecycleContext& ctx) noexcept {
  return impl_->guards.run<holoscan::LifecycleStage::kArm>([this, &ctx]() noexcept {
    auto sender = ctx.notification_sender(frame_ready_);
    if (!sender) return stage_failure("arm", "notification sender unavailable");
    impl_->sender = std::move(*sender);
    return holoscan::LifecycleStatus::kOk;
  });
}

holoscan::LifecycleStatus ArgusCaptureOp::on_start(holoscan::LifecycleContext&) noexcept {
  return impl_->guards.run<holoscan::LifecycleStage::kStart>([this]() noexcept {
    try {
      impl_->sequence.reset();
      impl_->clock.unbind();
      impl_->failure = {};
      impl_->failure_code = holoscan::ErrorCode::kFailure;
      impl_->queue_drops = 0;
      impl_->timeouts = 0;
      impl_->publication_drops = 0;
      impl_->service->start();
      impl_->worker = std::jthread([this](const std::stop_token& stop) { impl_->read(stop); });
      return holoscan::LifecycleStatus::kOk;
    } catch (const std::exception& e) {
      try {
        impl_->stop();
      } catch (...) {
      }
      return stage_failure("start", e.what());
    } catch (...) {
      try {
        impl_->stop();
      } catch (...) {
      }
      return stage_failure("start", "unknown backend failure");
    }
  });
}

holoscan::LifecycleStatus ArgusCaptureOp::on_stop(holoscan::LifecycleContext&) noexcept {
  return impl_->guards.run<holoscan::LifecycleStage::kStop>([this]() noexcept {
    try {
      impl_->stop();
      return holoscan::LifecycleStatus::kOk;
    } catch (const std::exception& e) {
      return stage_failure("stop", e.what());
    } catch (...) {
      return stage_failure("stop", "unknown backend failure");
    }
  });
}

holoscan::LifecycleStatus ArgusCaptureOp::on_release(holoscan::LifecycleContext&) noexcept {
  return impl_->guards.run<holoscan::LifecycleStage::kRelease>([this]() noexcept {
    // Keep release reachable after a failed stop and make worker termination unconditional.
    auto status = holoscan::LifecycleStatus::kOk;
    try {
      impl_->stop();
    } catch (const std::exception& e) {
      status = stage_failure("release", e.what());
    } catch (...) {
      status = stage_failure("release", "unknown backend failure");
    }
    impl_->service.reset();
    impl_->clock.unbind();
    impl_->sender = {};
    return status;
  });
}

holoscan::expected<void, holoscan::Error> ArgusCaptureOp::compute(
    holoscan::ExecutionContext& context) {
  std::optional<argus::CapturedFrame> captured;
  try {
    {
      std::lock_guard lock(impl_->mutex);
      impl_->notification_pending = false;
      if (impl_->failure) std::rethrow_exception(impl_->failure);
      captured = impl_->queue.pop();
    }
    if (!captured) return {};
    const std::array<std::int64_t, 1> shape{static_cast<std::int64_t>(frame_bytes_)};
    // The SDK returns a TensorOutputLoan for runtime-owned tensor storage. Fill and commit
    // through its write guard before emit_tensor() consumes the loan for publication.
    auto tensor_loan =
        frame.allocate_tensor(holoscan::TensorLoanRequest{.shape = shape, .dtype = kByte});
    if (!tensor_loan) {
      const auto slot = captured->slot;
      captured.reset();
      impl_->service->release(slot);
      if (dropped_publication(tensor_loan.error().code)) {
        ++impl_->publication_drops;
        return {};
      }
      return holoscan::make_unexpected(std::move(tensor_loan).error());
    }
    const auto placement = tensor_loan->memory_kind();
    const bool host =
        placement == holoscan::MemoryKind::kHost || placement == holoscan::MemoryKind::kPinnedHost;
    auto writer = host ? tensor_loan->write_host() : tensor_loan->write(context.cuda_stream());
    if (!writer) throw std::runtime_error("HSDK tensor writer unavailable");
    impl_->service->copy_frame(captured->slot, *writer, placement);
    // No native buffer escapes into the graph. Release only after copy completion.
    auto data = std::move(captured->metadata);
    const auto slot = captured->slot;
    captured.reset();
    impl_->service->release(slot);
    auto committed = std::move(*writer).commit();
    if (!committed) return holoscan::make_unexpected(std::move(committed).error());

    holoscan::EmitOptions options;
    options.frame_id = data.sequence;
    const auto observation = impl_->sequence.observe(data.sequence);
    data.sequence_gap = observation.gap;
    data.sequence_regressed = observation.regressed;
    data.queue_drops = impl_->queue_drops.load();
    data.publication_drops = impl_->publication_drops;
    data.capture_timeouts = impl_->timeouts.load();
    if (observation.gap || observation.regressed) options.flags |= holoscan::SampleFlags::kDegraded;

    if (config_.integration_start_offset_ns &&
        data.timestamp_clock == ArgusTimestampClock_TEGRA_TSC_NS) {
      if (!impl_->clock.settled()) {
        const auto reference = context.clock();
        impl_->clock.bind(reference);
        const auto status = impl_->clock.prime(
            [this]() noexcept { return impl_->service->timestamp_now_ns(); },
            [&reference]() noexcept -> std::optional<std::int64_t> {
              auto now = reference.now();
              return now ? std::optional<std::int64_t>{now->timestamp_ns} : std::nullopt;
            });
        if (status != holoscan::LifecycleStatus::kOk) {
          throw std::runtime_error("cannot project the camera timestamp into the HSDK clock");
        }
      }
      const auto raw = data.sensor_sof_timestamp_ns;
      const auto offset = *config_.integration_start_offset_ns;
      constexpr auto limit = std::numeric_limits<std::int64_t>::max();
      if (raw > static_cast<std::uint64_t>(limit) ||
          (offset > 0 && raw > static_cast<std::uint64_t>(limit - offset))) {
        throw std::runtime_error("Argus capture timestamp overflow");
      }
      const auto corrected = static_cast<std::int64_t>(raw) + offset;
      options.capture_time = impl_->clock.project(corrected);
      if (!options.capture_time) throw std::runtime_error("Argus timestamp projection overflow");
      data.capture_timestamp_ns = options.capture_time->timestamp_ns;
      data.capture_time_valid = true;
      data.clock_uncertainty_ns = impl_->clock.estimate().uncertainty_ns;
    }

    holoscan::schema::ImageT image;
    image.width = static_cast<std::int32_t>(config_.width);
    image.height = static_cast<std::int32_t>(config_.height);
    image.encoding = config_.pixel_format == ArgusPixelFormat::kNv12
                         ? holoscan::schema::ImageEncoding_NV12
                         : holoscan::schema::ImageEncoding_I420;
    image.header = std::make_shared<holoscan::schema::HeaderT>();
    image.header->frame_id = config_.frame_id;
    image.header->device_sequence = data.sequence;
    image.header->capture_timestamp_ns = data.capture_timestamp_ns;
    image.exposure_time_ns = static_cast<std::int64_t>(data.exposure_time_ns);
    image.gain = data.analog_gain * data.isp_digital_gain;
    const std::uint64_t y_bytes = static_cast<std::uint64_t>(config_.width) * config_.height;
    image.plane_layouts.emplace_back(0, config_.width, y_bytes);
    if (config_.pixel_format == ArgusPixelFormat::kNv12) {
      image.plane_layouts.emplace_back(y_bytes, config_.width, y_bytes / 2);
    } else {
      image.plane_layouts.emplace_back(y_bytes, config_.width / 2, y_bytes / 4);
      image.plane_layouts.emplace_back(y_bytes + y_bytes / 4, config_.width / 2, y_bytes / 4);
    }
    auto emitted = frame.emit_tensor(std::move(*tensor_loan), image, options);
    if (!emitted) {
      if (!dropped_publication(emitted.error().code)) return emitted;
      ++impl_->publication_drops;
      // Some fan-out branches may have accepted an indeterminate publication. Never retry it.
      if (emitted.error().code != holoscan::ErrorCode::kPublicationIndeterminate) return {};
    }
    auto stamped = metadata.emit(data, options);
    if (!stamped) {
      if (!dropped_publication(stamped.error().code)) return stamped;
      ++impl_->publication_drops;
    }
    return {};
  } catch (const std::exception& e) {
    if (captured) {
      try {
        impl_->service->release(captured->slot);
      } catch (...) {
      }
    }
    return holoscan::make_unexpected(holoscan::Error{impl_->failure_code.load(), e.what()});
  }
}
}  // namespace holoscan::holoscan_camera
