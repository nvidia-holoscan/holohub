// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "v4l2_capture_op/v4l2_capture_op.hpp"

// clang-format off: see the note on the include block in v4l2_capture_op.hpp.
#include <algorithm>
#include <array>
#include <chrono>  // NOLINT(build/c++11)
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

#include <cuda/stream>  // NOLINT(build/include_order)
#include <cuda_runtime.h>  // NOLINT(build/include_order)
#include <holoscan/logger/logger.hpp>

#include "v4l2_capture_op/v4l2_device.hpp"
// clang-format on

namespace holoscan::holoscan_camera {
namespace {

constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
constexpr std::uint32_t kCaptureBuffers = 4U;
constexpr std::uint32_t kCaptureBytesPerPixel = 2U;  // YUYV 4:2:2

// How long the reader thread blocks in poll() before checking its stop token. Long enough that a
// stalled device does not spin, short enough that a stop request is honoured promptly.
constexpr std::chrono::milliseconds kReaderPollTimeout{100};
// Consecutive failed waits after which the device is treated as lost rather than unlucky. Stated
// as a constant because it is a policy rather than a consequence of control flow: a descriptor in
// POLLERR or POLLNVAL answers immediately and keeps answering, so this is not a device that might
// yet produce a frame, and the tolerance exists only so a single anomaly does not end a run. At
// kReaderPollTimeout per attempt this declares loss within a few seconds, deliberately inside the
// reference application's default frame timeout: a lost device should fail the run by name rather
// than time out waiting for frames and name nothing.
constexpr std::uint32_t kReaderFailureTolerance = 32U;
// Backoff when a notification cannot be delivered yet. Both reasons for that -- the graph has not
// reached its global start barrier, and a prior activation is still retiring -- are bounded states
// rather than failures.
constexpr std::chrono::milliseconds kPostRetryInterval{1};
// How often a clock discipline that will not settle is allowed to say so. Priming is reattempted on
// every activation until it succeeds, so an unthrottled complaint is one log line per frame; at a
// few hundred frames this is a line every several seconds, which is frequent enough to follow and
// sparse enough to leave the log usable.
constexpr std::uint32_t kClockPrimeComplaintInterval = 256U;
// How often a device that keeps refusing is allowed to say so. Shorter than the clock interval
// because a refusing device is a fault rather than a device that merely will not bracket tightly,
// and the operator has nothing else to report while it lasts.
constexpr std::uint32_t kDeviceFailureComplaintInterval = 64U;
// How often a counter that will not advance is allowed to say so. Matched to the device failure
// interval because a regression is a fault of the same kind, but it cannot be assumed rare: a
// driver that leaves the counter at zero regresses on every frame after the first, and the interval
// is what keeps that case from burying the rest of the log.
constexpr std::uint32_t kSequenceRegressionComplaintInterval = 64U;
// How often a timestamp whose source is not the start of integration is allowed to say so.
// Throttled on the same reasoning as the regression interval and for a stronger reason: most
// drivers stamp end-of-frame for every frame they ever deliver, so this is a fixed device property
// rather than an intermittent fault, and it would otherwise log once per frame for a whole run.
constexpr std::uint32_t kDeclinedTimestampComplaintInterval = 256U;

/// @brief One producer's view of the bytes behind a frame.
/// @note There is no stride. The published frame is the captured frame, so the copy is contiguous;
/// a stride existed only to walk past the chroma that is now carried too.
struct CapturedPixels {
  const std::uint8_t* data{};  ///< First byte of the producer's buffer.
  std::size_t bytes{};         ///< Bytes the producer actually wrote, which may be short.
};

/// @brief Log a device failure from a `noexcept` context.
///
/// Formatting can throw, and every caller here is either a lifecycle callback or the reader thread,
/// both of which are `noexcept`. An escaping exception would terminate the process over a log line.
void log_failure(std::string_view stage, const v4l2::Failure& failure) noexcept {
  try {
    HOLOSCAN_LOG_ERROR("v4l2 capture failed during {}: {} (errno {})", stage, failure.op,
                       failure.error);
  } catch (...) {
  }
}

/// @brief Log a failure that has no `errno` behind it, from a `noexcept` context.
/// @see log_failure for why the catch is empty.
void log_message(std::string_view message) noexcept {
  try {
    HOLOSCAN_LOG_ERROR("v4l2 capture: {}", message);
  } catch (...) {
  }
}

/// @brief Report a frame that could not be published, from a `noexcept` context.
/// @param error The error the publication returned.
/// @see log_failure for why the catch is empty.
void log_publication_failure(const holoscan::Error& error) noexcept {
  try {
    HOLOSCAN_LOG_ERROR("v4l2 capture could not publish a frame: {}",
                       holoscan::runtime_error_message(error));
  } catch (...) {
  }
}

/// @brief Report a driver that refused to hand over a completed buffer, from a `noexcept` context.
/// @param failure The operation and `errno` the device recorded for the refusal.
/// @param attempts Consecutive refusals, including this one.
/// @see log_failure for why the catch is empty.
void log_dequeue_failure(const v4l2::Failure& failure, std::uint32_t attempts) noexcept {
  try {
    HOLOSCAN_LOG_ERROR("v4l2 capture could not take a completed buffer: {} (errno {}), {} in a row",
                       failure.op, failure.error, attempts);
  } catch (...) {
  }
}

/// @brief Report a device counter that moved backwards or repeated, from a `noexcept` context.
/// @param sequence The counter value that arrived out of order.
/// @param previous The counter value observed before it.
/// @param occurrences Consecutive regressions, including this one.
/// @see log_failure for why the catch is empty.
/// @note Both values are reported because the difference between them is what names the cause: a
/// fall to zero is a counter the driver restarted, a fall from near a power of two is a counter
/// that wrapped narrower than the tracker reads, and an unchanged value is a driver that never
/// filled the field at all.
void log_sequence_regression(std::uint64_t sequence, std::uint64_t previous,
                             std::uint32_t occurrences) noexcept {
  try {
    HOLOSCAN_LOG_WARN(
        "v4l2 capture: device sequence went from {} to {}, so samples across the discontinuity "
        "cannot be counted, {} in a row",
        previous, sequence, occurrences);
  } catch (...) {
  }
}

/// @brief Name a timestamp source for a diagnostic.
[[nodiscard]] std::string_view timestamp_source_name(v4l2::TimestampSource source) noexcept {
  switch (source) {
    case v4l2::TimestampSource::kEndOfFrame:
      return "end-of-frame";
    case v4l2::TimestampSource::kStartOfExposure:
      return "start-of-exposure";
    default:
      return "unstated";
  }
}

/// @brief Report a timestamp this operator declined to publish as a capture time.
/// @param source What instant of the frame the driver's timestamp actually marked.
/// @param occurrences Consecutive frames whose timestamp was declined, including this one.
/// @see log_failure for why the catch is empty.
/// @note Worth saying out loud rather than silently omitting the field, because the two readings of
/// an absent capture time are far apart: a driver that names no clock cannot be projected at all,
/// while this one hands over a perfectly good coordinate that simply marks the wrong instant. The
/// remedy differs too -- the first needs a different driver, this one needs `V4L2_BUF_FLAG_TSTAMP_
/// SRC_SOE` support or exposure metadata to correct by.
void log_declined_timestamp(v4l2::TimestampSource source, std::uint32_t occurrences) noexcept {
  try {
    HOLOSCAN_LOG_WARN(
        "v4l2 capture: the driver stamps {}, which is not the start of integration the capture "
        "time means, so the frame is published without one, {} in a row",
        timestamp_source_name(source), occurrences);
  } catch (...) {
  }
}

/// @brief Report a descriptor that will not become readable, from a `noexcept` context.
/// @param wait The failed wait, carrying either an `errno` or the conditions `poll` reported.
/// @param attempts Consecutive failed waits, including this one.
/// @see log_failure for why the catch is empty.
void log_wait_failure(const v4l2::Wait& wait, std::uint32_t attempts) noexcept {
  try {
    HOLOSCAN_LOG_ERROR(
        "v4l2 capture cannot wait on the device: errno {}, poll revents {:#x}, "
        "{} in a row",
        wait.error, wait.revents, attempts);
  } catch (...) {
  }
}

/// @brief Pack a failed wait into one word so it can cross to the operator through an atomic.
/// @param wait The failed wait to encode.
/// @return A nonzero encoding of the failure.
/// @note Zero is reserved for "no failure recorded", which costs nothing: a failed wait always
/// carries either an `errno` or a nonzero `revents`, so no genuine failure encodes to zero. The
/// pair travels together because either one alone names a different cause -- an `errno` is `poll`
/// itself refusing, while `revents` is `poll` succeeding and reporting the descriptor is broken.
[[nodiscard]] constexpr std::uint64_t encode_wait_failure(const v4l2::Wait& wait) noexcept {
  return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(wait.error)) << 32U) |
         static_cast<std::uint64_t>(static_cast<std::uint32_t>(wait.revents));
}

/// @brief Report that the reader gave up on a descriptor, from a `noexcept` context.
/// @param encoded The packed failure the reader recorded.
/// @see log_failure for why the catch is empty.
void log_device_lost(std::uint64_t encoded) noexcept {
  try {
    HOLOSCAN_LOG_ERROR(
        "v4l2 capture: the device stopped becoming readable (errno {}, poll events {:#x}) for {} "
        "consecutive waits, so the run is failed",
        static_cast<std::int32_t>(encoded >> 32U),
        static_cast<std::uint32_t>(encoded & 0xFFFFFFFFU), kReaderFailureTolerance);
  } catch (...) {
  }
}

/// @brief Report that the clock discipline declined to settle, from a `noexcept` context.
/// @param status The non-settling status \ref holoscan::sensor_io::ClockDiscipline::prime returned.
/// @param attempts Consecutive priming attempts that have now failed, including this one.
/// @see log_failure for why the catch is empty.
/// @note The measured uncertainty is deliberately not reported. A rejected prime does not store its
/// estimate, so reading it back would print the previous epoch's figure or a zero, and a wrong
/// number here is worse than none: it is the number a reader would use to decide the clocks are
/// fine.
void log_clock_unsettled(holoscan::LifecycleStatus status, std::uint32_t attempts) noexcept {
  try {
    if (status == holoscan::LifecycleStatus::kRetryableFailure) {
      HOLOSCAN_LOG_WARN(
          "v4l2 capture: no clock measurement bracketed tighter than {} ns in {} attempt(s), so "
          "capture timestamps stay absent until one does",
          holoscan::sensor_io::ClockDiscipline::kDefaultMaxUncertaintyNs, attempts);
      return;
    }
    HOLOSCAN_LOG_ERROR(
        "v4l2 capture: no paired clock read succeeded in {} attempt(s), so capture timestamps "
        "stay absent",
        attempts);
  } catch (...) {
  }
}

/// @brief The two spans a fill writes into one allocation.
struct FillSpans {
  std::uint8_t* destination{};  ///< First byte of the allocation.
  std::size_t copied{};         ///< Bytes to take from the driver, clamped to the allocation.
  std::size_t zeroed{};         ///< Bytes after them that the driver did not fill.
};

/// @brief Decide what a fill writes, without writing any of it.
/// @param writer Guard over the allocation being filled.
/// @param pixels Producer buffer and the bytes it actually wrote.
/// @return The spans to copy and to zero, or a rejection when there is nothing to write between.
/// @note Shared by the host and device fills so that the clamp and the size of the remainder are
/// decided in one place. The two paths differ in the primitive they write with and in whether that
/// primitive can fail, and only that difference belongs in each of them; a rule duplicated between
/// them is a rule that can come to mean two things without anything reporting that it has.
/// @note The remainder is zeroed rather than left as it was found, so a short frame at least
/// fabricates the same rows every time. The caller has already marked the sample kDegraded, which
/// is what tells a consumer those rows are not measurements.
[[nodiscard]] holoscan::expected<FillSpans, holoscan::Error> plan_fill(
    const holoscan::TensorOutputWriteGuard& writer, const CapturedPixels& pixels) {
  auto* destination = writer.data_as<std::uint8_t>();
  if (destination == nullptr || pixels.data == nullptr) {
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kInvalidArgument});
  }
  // A driver reports the bytes it wrote, which may be fewer than the allocation and, for a driver
  // that misreports, more. Clamping both ways is what keeps the copy inside both buffers.
  const std::size_t copied = std::min(writer.byte_size(), pixels.bytes);
  return FillSpans{
      .destination = destination, .copied = copied, .zeroed = writer.byte_size() - copied};
}

/// @brief Publish one captured frame as a descriptor-backed holoscan::schema::ImageT.
/// @param port Output port to allocate the tensor from.
/// @param pixels Producer buffer and the bytes it actually wrote.
/// @param width Pixels per row of the published frame.
/// @param height Rows in the published frame.
/// @param frame_of_reference Coordinate frame the image is expressed in.
/// @param options Metadata the producer established for this sample.
/// @param stream Activation stream to order a device-placed write on; unused for a host placement.
/// @return Success, or the first allocation, write, or emit failure.
/// @note A free function rather than a member so the operator's header does not have to expose the
/// producer-side buffer view. Publication is the same act for any source of packed 4:2:2 frames;
/// only the metadata in \p options is device-specific.
[[nodiscard]] holoscan::expected<void, holoscan::Error> publish_frame(
    holoscan::Output<holoscan::schema::ImageT>& port,  // NOLINT(runtime/references)
    const CapturedPixels& pixels, int width, int height, std::string_view frame_of_reference,
    const holoscan::EmitOptions& options, ::cuda::stream_ref stream) {
  // A row is 2 * width bytes, which is what the even width the constructor insists on makes equal
  // to the 4 * ceil(width / 2) that ImageEncoding_YUYV derives. The Tensor therefore addresses
  // exactly the frame the descriptor describes, with nothing spare for the schema to reject.
  const std::array<std::int64_t, 2U> shape{
      height, static_cast<std::int64_t>(width) * kCaptureBytesPerPixel};
  auto loan = port.allocate_tensor(
      holoscan::TensorLoanRequest{.shape = shape, .dtype = kFrameElementDtype});
  if (!loan) {
    return holoscan::make_unexpected(std::move(loan).error());
  }
  // Branch on the placement the compiled plan froze rather than on what the operator asked for.
  // They agree, but only one of them is what the allocation actually is, and write_host() is
  // admitted only for host-accessible storage.
  const holoscan::MemoryKind placement = loan->memory_kind();
  const bool host_addressable =
      placement == holoscan::MemoryKind::kHost || placement == holoscan::MemoryKind::kPinnedHost;

  // Each fill decides nothing of its own: plan_fill clamps and sizes the remainder for both, so the
  // one rule cannot come to mean two things. What is left in each is the primitive its placement
  // requires, and for the device the failures that primitive can report.
  const auto host_fill = [&pixels](const holoscan::TensorOutputWriteGuard& writer)
      -> holoscan::expected<void, holoscan::Error> {
    auto spans = plan_fill(writer, pixels);
    if (!spans) {
      return holoscan::make_unexpected(std::move(spans).error());
    }
    std::memcpy(spans->destination, pixels.data, spans->copied);
    if (spans->zeroed != 0U) {
      std::memset(spans->destination + spans->copied, 0, spans->zeroed);
    }
    return {};
  };

  const auto device_fill = [&pixels, stream](const holoscan::TensorOutputWriteGuard& writer)
      -> holoscan::expected<void, holoscan::Error> {
    auto spans = plan_fill(writer, pixels);
    if (!spans) {
      return holoscan::make_unexpected(std::move(spans).error());
    }
    // The source is the driver's pageable mmap buffer. For a pageable source cudaMemcpyAsync does
    // not return until those bytes have been staged, so the caller may requeue the buffer to the
    // driver as soon as this returns even though the DMA to the device has not finished. That is
    // what makes the requeue in compute() safe today, and it is exactly the guarantee that would
    // be lost if this capture path ever moved to USERPTR or DMABUF buffers -- at which point the
    // requeue would have to wait on the stream instead.
    if (cudaMemcpyAsync(spans->destination, pixels.data, spans->copied, cudaMemcpyHostToDevice,
                        stream.get()) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    if (spans->zeroed != 0U && cudaMemsetAsync(spans->destination + spans->copied, 0, spans->zeroed,
                                               stream.get()) != cudaSuccess) {
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    return {};
  };

  auto committed = host_addressable ? loan->write_host_and_commit(host_fill)
                                    : loan->write_and_commit(stream, device_fill);
  if (!committed) {
    return holoscan::make_unexpected(std::move(committed).error());
  }

  holoscan::schema::ImageT descriptor;
  descriptor.width = width;
  descriptor.height = height;
  descriptor.encoding = holoscan::schema::ImageEncoding_YUYV;
  descriptor.header = std::make_shared<holoscan::schema::HeaderT>();
  // Header::frame_id names the coordinate frame; EmitOptions::frame_id is the numeric sample
  // identity. They are unrelated fields that happen to share a name.
  descriptor.header->frame_id = std::string{frame_of_reference};
  if (options.frame_id.has_value()) {
    descriptor.header->device_sequence = *options.frame_id;
  }
  // Only acquisition time is written into the payload, and it is the same value carried in
  // EmitOptions::capture_time rather than a second reading, so the two can never disagree.
  //
  // Header carries no publication time at all, because publication belongs to the transport
  // rather than the payload: a payload copy would be a second answer to one question that a
  // producer could set inconsistently. The runtime stamps publish_timestamp_ns and the
  // publish_clock_id naming its domain onto SampleMetadata when it submits the sample, so a
  // consumer that wants publication time reads it from the envelope and nothing here should
  // duplicate it. Acquisition time is the one a recording cannot reconstruct, which is why it
  // travels in the payload instead.
  if (options.capture_time.has_value()) {
    descriptor.header->capture_timestamp_ns = options.capture_time->timestamp_ns;
  }
  return port.emit_tensor(std::move(*loan), descriptor, options);
}

}  // namespace

V4l2CaptureOp::V4l2CaptureOp(std::string device, int width, int height, int fps,
                             std::string frame_id, holoscan::MemoryKind memory_kind)
    : device_(std::move(device)),
      frame_id_(std::move(frame_id)),
      width_(width),
      height_(height),
      fps_(fps),
      memory_kind_(memory_kind),
      device_handle_(std::make_unique<v4l2::Device>()) {
  if (device_.empty()) {
    throw std::invalid_argument("V4L2 device path must not be empty");
  }
  if (width_ <= 0 || height_ <= 0 || fps_ <= 0) {
    throw std::invalid_argument("V4L2 width, height, and fps must all be positive");
  }
  if (fps_ > kNanosecondsPerSecond) {
    throw std::invalid_argument("V4L2 fps exceeds the nanosecond clock resolution");
  }
  if ((width_ & 1) != 0) {
    // A packed 4:2:2 group spans two columns, so an odd width leaves a half group. The schema
    // rounds that group up and would require 4 * ceil(width / 2) bytes per row where the driver
    // delivers 2 * width, and the frame would be refused for being a row short of its own
    // description. Refusing the mode here reports the cause instead of the symptom.
    throw std::invalid_argument("YUYV capture width must be even");
  }
  if (frame_id_.empty()) {
    // An image whose coordinate frame is the empty string is indistinguishable from one whose frame
    // was never set, and a consumer cannot place it relative to any other sensor.
    throw std::invalid_argument("V4L2 frame_id must name a coordinate frame");
  }
  if (memory_kind_ != holoscan::MemoryKind::kHost &&
      memory_kind_ != holoscan::MemoryKind::kPinnedHost &&
      memory_kind_ != holoscan::MemoryKind::kCudaDevice) {
    // kUnknown is a diagnostic value the schema never admits for a published tensor. kCudaManaged
    // is host-writable and would appear to work, but a captured frame that migrates on first device
    // touch trades a copy this operator can see for a fault it cannot, so it is refused until some
    // consumer asks for it and can say why.
    throw std::invalid_argument("V4L2 memory kind must be kHost, kPinnedHost, or kCudaDevice");
  }

  const auto unsigned_width = static_cast<std::size_t>(width_);
  const auto unsigned_height = static_cast<std::size_t>(height_);
  if (unsigned_height != 0U && unsigned_width > std::numeric_limits<std::size_t>::max() /
                                                    unsigned_height / kCaptureBytesPerPixel) {
    throw std::invalid_argument("V4L2 frame byte count overflows size_t");
  }

  capture_bytes_ = unsigned_width * unsigned_height * kCaptureBytesPerPixel;
}

V4l2CaptureOp::~V4l2CaptureOp() = default;

void V4l2CaptureOp::setup(holoscan::OperatorSpec& spec) {
  // The descriptor's Tensor is two-dimensional -- rows by row bytes, one byte per element --
  // because the published payload is the packed YUYV frame and a row of it is 2 * width bytes. The
  // pixel structure within a row is the encoding's to state, not the shape's.
  spec.output(frame, "frame")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{
          .representation = {.memory_kind = memory_kind_, .dtype = kFrameElementDtype, .rank = 2U},
          .bounds = holoscan::tensor_bounds(capture_bytes_),
          .storage = holoscan::TensorOutputStorage::kRuntimePool,
      });
  // The reader thread owns the descriptor wait; this is how it wakes the operator.
  //
  // Two sender records rather than one, for the restart replay and not for concurrency: only the
  // reader ever posts. A restart advances the source generation, which retires the sender this
  // operator holds, and `on_arm` then has to acquire its replacement. Core hands out only records
  // no one has claimed, and the retired member still claims its own until the assignment below
  // completes -- so with a single record the acquisition fails with kResourceExhausted and the
  // restart that was supposed to restore capture is what breaks it. The spare is what the
  // predecessor occupies while its successor is being acquired. Core's own restart test configures
  // capacity two for this reason.
  spec.notification_source(frame_ready_, "v4l2-frame-ready")
      .capacity(1U)
      .sender_reference_capacity(2U);
  // Registered against this exact concrete type, which is what Core requires: a member pointer
  // from a base class is rejected when the graph validates the declaration. Every stage a capture
  // device owns is named here and none is declared optional, so a hook that is renamed or dropped
  // fails to compile rather than silently going unregistered.
  spec.lifecycle()
      .stage(holoscan::LifecycleStage::kConfigure, &V4l2CaptureOp::on_configure)
      .stage(holoscan::LifecycleStage::kDiscover, &V4l2CaptureOp::on_discover)
      .stage(holoscan::LifecycleStage::kAllocate, &V4l2CaptureOp::on_allocate)
      .stage(holoscan::LifecycleStage::kArm, &V4l2CaptureOp::on_arm)
      .stage(holoscan::LifecycleStage::kStart, &V4l2CaptureOp::on_start)
      .stage(holoscan::LifecycleStage::kStop, &V4l2CaptureOp::on_stop)
      .stage(holoscan::LifecycleStage::kRelease, &V4l2CaptureOp::on_release);
}

holoscan::Contract V4l2CaptureOp::contract() const {
  holoscan::Contract result;
  // A clock trigger on a capture source is a guess about when frames exist. This is not a guess:
  // the reader thread posts once per completed buffer.
  result.trigger(holoscan::OnNotified{.event = frame_ready_});
  return result;
}

holoscan::LifecycleStatus V4l2CaptureOp::on_configure(holoscan::LifecycleContext&) noexcept {
  return guards_.run<holoscan::LifecycleStage::kConfigure>([this]() noexcept {
    clock_.unbind();
    if (!device_handle_->open(device_.c_str())) {
      return device_failure("open");
    }
    return holoscan::LifecycleStatus::kOk;
  });
}

holoscan::LifecycleStatus V4l2CaptureOp::on_discover(holoscan::LifecycleContext&) noexcept {
  return guards_.run<holoscan::LifecycleStage::kDiscover>([this]() noexcept {
    // VIDIOC_S_FMT succeeds while quietly rounding the geometry, so the device reports what it
    // actually granted and this stage refuses anything that does not match the declaration in
    // setup(). A pipeline sized for one geometry that silently receives another produces corruption
    // attributed to whatever consumes it.
    const v4l2::Format desired{.code = v4l2::kFourccYuyv,
                               .width = static_cast<std::uint32_t>(width_),
                               .height = static_cast<std::uint32_t>(height_)};
    if (!device_handle_->negotiate(desired, static_cast<std::uint32_t>(fps_))) {
      return device_failure("negotiate");
    }
    // One size governs both sides now that the frame is forwarded whole: the same figure bounds the
    // driver buffer here and the Tensor declared in setup(). A driver asking for more than the
    // negotiated geometry needs is the one case the copy could not absorb without truncating.
    if (device_handle_->format().size_image > capture_bytes_) {
      log_message("driver frame size exceeds the negotiated capture size");
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  });
}

holoscan::LifecycleStatus V4l2CaptureOp::on_allocate(holoscan::LifecycleContext&) noexcept {
  return guards_.run<holoscan::LifecycleStage::kAllocate>([this]() noexcept {
    if (!device_handle_->map_buffers(kCaptureBuffers)) {
      return device_failure("map_buffers");
    }
    return holoscan::LifecycleStatus::kOk;
  });
}

holoscan::LifecycleStatus V4l2CaptureOp::on_arm(holoscan::LifecycleContext& context) noexcept {
  return guards_.run<holoscan::LifecycleStage::kArm>([this, &context]() noexcept {
    // kArm is not a choice. Core admits notification_sender() at kArm and nowhere else, and
    // rejects it from any other stage with kLifecycleInvalidCallingContext. This stage
    // deliberately owns nothing else: Core retains a completed kArm across a restart and runs the
    // bare suffix kStop, kStart for one requested at kStart, so anything acquired here that kStop
    // tears down would never be re-acquired.
    //
    // A restart advances the notification source generation for every vertex in its closure without
    // consulting the stage it was requested from, so the sender acquired here is retired by any
    // restart and has to be re-acquired. Replaying kArm is what does that, and the declaration's
    // spare sender record is what lets it: the acquisition below runs while the retired member
    // still claims its own record, and Core hands out only unclaimed ones.
    //
    // A restart requested at kStart is the case this cannot serve. Core replays the bare suffix
    // kStop, kStart and retains a completed kArm, so this stage does not run, the retired sender
    // stays in place, and the reader's first post fails with kStaleEndpoint. Closing that needs one
    // of two things from Core -- sender acquisition admitted at kStart, or a notification
    // generation that advances only for restarts widened to kArm -- and neither is reachable from
    // an operator, because widening is driven by lifecycle dependency edges between two operators
    // and a source cannot declare its own floor. Until then the reader names the cause when it
    // happens rather than going quiet, and a graph that restarts this operator should request kArm
    // or earlier.
    auto sender = context.notification_sender(frame_ready_);
    if (!sender) {
      log_message("the notification sender could not be acquired at kArm");
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    // Assigned over the predecessor, which releases the record it held. Ordering matters on a
    // replay: the acquisition above must complete first, so the two records are briefly both
    // claimed and the spare is what makes that legal.
    sender_ = std::move(*sender);
    return holoscan::LifecycleStatus::kOk;
  });
}

holoscan::LifecycleStatus V4l2CaptureOp::on_start(holoscan::LifecycleContext&) noexcept {
  return guards_.run<holoscan::LifecycleStage::kStart>([this]() noexcept {
    // Stream activation belongs here rather than at kArm because kStop undoes it. kStart is the
    // minimal forward suffix Core replays, so it has to restore everything a stop took away or a
    // start-only restart resumes a reader against a device that is no longer streaming.
    //
    // The sequence reset is paired with stream_on for the same reason it always was: the counter
    // belongs to the stream and VIDIOC_STREAMON restarts it, so resetting anywhere the restart
    // suffix might skip would leave the tracker holding a pre-restart sequence and report the
    // driver's fresh zero as a regression.
    sequence_.reset();
    // The regression count belongs to the tracker it throttles, so it is cleared with it. Carrying
    // it across would leave the new stream's first regression mid-interval and silence it.
    sequence_regressions_ = 0U;
    // Cleared for the same reason, and it is a per-stream property in its own right: a driver may
    // state a different timestamp source after a STREAMON, so the new stream's first declined
    // frame should report rather than land mid-interval and stay quiet.
    declined_timestamps_ = 0U;
    // The post interlock belongs to the stream for the same reason, and carrying it across is
    // unrecoverable rather than merely stale. The reader raises it and only compute() lowers it, so
    // a stop landing between the two -- a buffer claimed that no activation dequeued, or a post
    // still retrying when the stop token tripped -- leaves it raised with no activation left to
    // lower it. The operator's only trigger is the notification the raised interlock now stops the
    // reader from posting, so the next stream's reader spins on it, compute() is never called, and
    // capture never resumes. kStop joined the previous reader before this runs, so nothing else can
    // observe the store.
    pending_.store(false, std::memory_order_release);
    // Cleared with the interlock, and for the matching reason: it belongs to the reader the next
    // line is about to replace. A failure the previous stream's reader recorded and no activation
    // consumed -- the graph stopped before one ran -- would otherwise fail the resumed stream's
    // first activation for a device that is, as far as this stream knows, working.
    reader_failure_.store(0U, std::memory_order_release);
    if (!device_handle_->stream_on()) {
      return device_failure("stream_on");
    }
    try {
      reader_ = std::jthread(
          [this](const std::stop_token& stop_token) noexcept { reader_loop(stop_token); });
    } catch (...) {
      // The stream was turned on by this body, so this body hands it back rather than leaving a
      // device producing frames that nothing will ever dequeue.
      static_cast<void>(device_handle_->stream_off());
      log_message("the reader thread could not be started");
      return holoscan::LifecycleStatus::kFatalFailure;
    }
    return holoscan::LifecycleStatus::kOk;
  });
}

holoscan::LifecycleStatus V4l2CaptureOp::on_stop(holoscan::LifecycleContext&) noexcept {
  return guards_.run<holoscan::LifecycleStage::kStop>([this]() noexcept {
    // The thread is owned by the lifecycle and is never detached: request, then join, then quiesce
    // the device.
    if (reader_.joinable()) {
      reader_.request_stop();
      try {
        reader_.join();
      } catch (...) {
        log_message("the reader thread could not be joined");
        return holoscan::LifecycleStatus::kFatalFailure;
      }
    }
    if (!device_handle_->stream_off()) {
      return device_failure("stream_off");
    }
    return holoscan::LifecycleStatus::kOk;
  });
}

holoscan::LifecycleStatus V4l2CaptureOp::on_release(holoscan::LifecycleContext&) noexcept {
  const holoscan::LifecycleStatus status = guards_.run<holoscan::LifecycleStage::kRelease>(
      [this]() noexcept { return release_resources(); });
  if (!guards_.guard_skipped(holoscan::LifecycleStage::kRelease)) {
    return status;
  }
  // The guard withheld the body, because it gates kRelease on kAllocate's ownership flag and that
  // flag stands for the mapped buffers alone. No flag it keeps has ever stood for the descriptor,
  // which on_configure opened two stages earlier, so every bring-up that fails before kAllocate
  // claims anything -- Configure succeeds, Discover rejects the geometry, kAllocate never runs --
  // reaches here with the device still open and is told there is nothing to release.
  //
  // Leaving it there would make the descriptor outlive the epoch that took it. ~Device() does
  // close it, but a destructor answers process exit, not lifecycle teardown: an operator that
  // failed bring-up should hand the node back before the graph tears down or retries, not when the
  // object dies. Until then a retry re-opens a node this instance still holds, and nothing else can
  // claim it.
  //
  // So the descriptor is released on the guard's skip as well, gated on the device itself rather
  // than on a flag that was never about it. That keeps teardown reachable after any bring-up
  // failure instead of only after one that got as far as kAllocate, and it is honest about what the
  // guard's flags mean rather than widening one to cover a resource it does not describe.
  if (!device_handle_->is_open()) {
    // Nothing kConfigure took is still held -- kConfigure itself failed, or a previous kRelease
    // already ran -- so the guard's skip is the accurate answer and there is no work to redo.
    return status;
  }
  return release_resources();
}

holoscan::LifecycleStatus V4l2CaptureOp::release_resources() noexcept {
  // Every line releases something a forward stage acquired, and nothing here is about the stream.
  // All three are idempotent, which is what lets this run either as the guarded kRelease body or as
  // the descriptor-only teardown its skip would otherwise strand. The sender is already dead by now
  // -- the runtime cancels readiness sources and closes notifications before kStop -- so dropping
  // it is hygiene rather than the act that stops the wakeups.
  sender_ = holoscan::NotificationSender{};
  device_handle_->close();
  // The measured offset relates this device's clock to the reference clock, and it is valid for
  // exactly as long as the device stays open. Closing it ends that relationship, so the next
  // generation must measure again rather than project through a stale one.
  clock_.unbind();
  return holoscan::LifecycleStatus::kOk;
}

holoscan::LifecycleStatus V4l2CaptureOp::device_failure(std::string_view stage) noexcept {
  log_failure(stage, device_handle_->last_error());
  return holoscan::LifecycleStatus::kFatalFailure;
}

void V4l2CaptureOp::ensure_clock_bound(holoscan::ExecutionContext& context) noexcept {
  if (clock_.settled()) {
    return;
  }
  const holoscan::ClockRef reference = context.clock();
  if (!reference.id().valid()) {
    return;
  }
  clock_.bind(reference);
  // V4L2 stamps buffers on CLOCK_MONOTONIC, which is a real clock and not the graph's. A handful of
  // paired reads removes a fixed bias from every subsequent frame.
  //
  // The status is read rather than discarded. prime() refuses to settle when even its tightest
  // bracket is wider than the maximum uncertainty it admits, and a refusal leaves the discipline
  // unsettled, so project() keeps reporting absence and every frame published until it settles
  // carries no capture time at all. That is the right behaviour and the wrong thing to stay quiet
  // about: absent capture times are indistinguishable from a driver that never named its clock, so
  // without this the operator's most consequential degradation is the one it never mentions.
  //
  // The bound stays at the default rather than a tighter figure named here. It is already enforced,
  // and choosing a narrower one is a statement about a consumer's alignment budget that a source
  // does not get to make. A device stamping the host's own CLOCK_MONOTONIC brackets orders of
  // magnitude inside the default, so what this rejects was preempted in every round rather than
  // merely imprecise -- which is also why the next activation is worth another attempt.
  const holoscan::LifecycleStatus primed =
      clock_.prime([]() noexcept { return v4l2::Device::monotonic_now_ns(); },
                   [&reference]() noexcept -> std::optional<std::int64_t> {
                     const auto now = reference.now();
                     if (!now) {
                       return std::nullopt;
                     }
                     return now->timestamp_ns;
                   });
  if (primed == holoscan::LifecycleStatus::kOk || primed == holoscan::LifecycleStatus::kSkipped) {
    clock_prime_failures_ = 0U;
    return;
  }
  // Core classifies kRetryableFailure but does not retry a stage, and this helper is not a stage: it
  // runs from every activation until the discipline settles, so the retry is already here and the
  // complaint is what needs bounding. Reporting the first failure always, and one in every
  // kClockPrimeComplaintInterval after it, keeps a device that never brackets cleanly from logging
  // once per frame while still leaving the condition visible for as long as it lasts.
  ++clock_prime_failures_;
  if (clock_prime_failures_ % kClockPrimeComplaintInterval == 1U) {
    log_clock_unsettled(primed, clock_prime_failures_);
  }
}

void V4l2CaptureOp::reader_loop(const std::stop_token& stop_token) noexcept {
  // Local rather than a member because it counts one stream's failures and the thread is created
  // fresh at every kStart, so the count starts over exactly when the stream does.
  std::uint32_t wait_failures = 0U;
  // The interlock below is waited on rather than polled, and a wait on an atomic ends only when the
  // value it was given changes. A stop request does not change it, so without this a stop arriving
  // while the interlock is raised would park this thread for good and kStop's join would never
  // return. Lowering it from the stop request is what releases the wait; the loop condition then
  // ends the thread. A bare notify would not do, because a woken waiter that finds the same value
  // waits again.
  //
  // Race-free in both directions: if this runs before the thread reaches the wait, the value the
  // wait would be given is already lowered, so it returns at once rather than losing the wake.
  const std::stop_callback release_interlock{stop_token, [this]() noexcept {
                                               pending_.store(false, std::memory_order_release);
                                               pending_.notify_all();
                                             }};
  while (!stop_token.stop_requested()) {
    // One post per completed buffer. Without this the descriptor stays readable until compute()
    // dequeues, and the loop would post repeatedly for a single frame.
    //
    // Waited on rather than polled. The interlock is raised for however long an activation takes to
    // reach the dequeue, which is the scheduler's business and not the device's, so a poll here
    // picks an interval that answers neither: too long and a frame that is already waiting is held
    // back by it, too short and every camera in a rig wakes to read a bool that has not changed.
    if (pending_.load(std::memory_order_acquire)) {
      pending_.wait(true, std::memory_order_acquire);
      continue;
    }
    const v4l2::Wait wait =
        device_handle_->wait_readable(static_cast<int>(kReaderPollTimeout.count()));
    if (wait.readiness == v4l2::Readiness::kFailed) {
      ++wait_failures;
      if (wait_failures % kDeviceFailureComplaintInterval == 1U) {
        log_wait_failure(wait, wait_failures);
      }
      // Past the tolerance the device is gone, and this thread cannot say so on its own: it holds
      // no activation to fail. Handing the cause to the operator and then posting is what turns a
      // lost device into a failed run -- the post buys the activation that returns the failure.
      // Recorded before the post so the activation it wakes cannot arrive first and find nothing.
      //
      // The loop ends here whether or not the post is accepted. A refused post means the graph is
      // already stopping or the sender is stale, and in both cases retrying a device that answers
      // only with errors serves no one.
      if (wait_failures >= kReaderFailureTolerance) {
        reader_failure_.store(encode_wait_failure(wait), std::memory_order_release);
        static_cast<void>(post_until_accepted(stop_token));
        return;
      }
      // Short of that, the descriptor answers poll immediately and keeps answering, so the timeout
      // that paces a quiet device does not pace this one. Waiting the span the poll would have
      // blocked for keeps a device that is gone costing what a device with nothing to say costs.
      std::this_thread::sleep_for(kReaderPollTimeout);
      continue;
    }
    if (wait.readiness != v4l2::Readiness::kReadable) {
      continue;
    }
    // Cleared on the first wait that succeeds, so a run of failures the device recovers from is
    // reported again if it returns, rather than being throttled against a stale count.
    wait_failures = 0U;
    pending_.store(true, std::memory_order_release);
    if (!post_until_accepted(stop_token)) {
      return;
    }
  }
}

bool V4l2CaptureOp::post_until_accepted(const std::stop_token& stop_token) noexcept {
  while (!stop_token.stop_requested()) {
    const auto posted = sender_.post();
    if (posted) {
      return true;
    }
    const holoscan::ErrorCode code = posted.error().code;
    // Before the global kStart barrier the source is not ready, and while a prior activation
    // retires it is backpressured. Both are bounded states, not failures.
    if (code == holoscan::ErrorCode::kNotReady || code == holoscan::ErrorCode::kBackpressured) {
      std::this_thread::sleep_for(kPostRetryInterval);
      continue;
    }
    // A closed source is the ordinary end of a stream: kStop closes it and joins this thread, so
    // leaving quietly is the correct behaviour and saying anything would make every clean shutdown
    // log.
    if (code == holoscan::ErrorCode::kClosed) {
      return false;
    }
    // A stale endpoint is not. It means a restart advanced this notification source's generation
    // and retired the authority kArm acquired, and the operator cannot obtain the replacement:
    // Core admits notification_sender() only at kArm, and a restart requested at kStart replays
    // kStop, kStart alone. The reader therefore ends here with no route back, the operator's only
    // trigger is gone, and capture stops for the remainder of the run while the lifecycle reports
    // the restart as successful. Nothing this thread can reach fails the activation -- the sender
    // was that route -- so the log is the whole diagnosis, and it has to name the cause rather
    // than let a camera go quiet for no stated reason. See the note in on_arm.
    if (code == holoscan::ErrorCode::kStaleEndpoint) {
      log_message(
          "a restart replaced this notification source generation and the reader cannot reacquire "
          "a sender outside kArm; this operator will publish no further frames");
      return false;
    }
    log_message("the reader thread could not post a frame notification");
    return false;
  }
  return false;
}

holoscan::expected<void, holoscan::Error> V4l2CaptureOp::compute(
    holoscan::ExecutionContext& context) {
  // This belongs in kWarmUp, which exists precisely so settling finishes before the operator
  // becomes activatable. It runs here because a lifecycle stage cannot reach a clock:
  // LifecycleContext::resolve_clock has a resolver slot that the runtime lifecycle executor never
  // installs, so it fails for every caller. ExecutionContext::clock() is populated, so the first
  // activation is the earliest point a real clock is reachable. Move this into on_warm_up once the
  // resolver is wired; nothing else about the operator changes.
  ensure_clock_bound(context);

  // Asked before the dequeue, because a device the reader has given up on has nothing to dequeue
  // and the failure is the answer. Taken rather than read, so the cause is reported once: this
  // returns a failure to Core, and Core is entitled to activate again before the graph unwinds.
  if (const std::uint64_t lost = reader_failure_.exchange(0U, std::memory_order_acq_rel);
      lost != 0U) {
    log_device_lost(lost);
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
  }

  const v4l2::Dequeued dequeued = device_handle_->dequeue();
  const std::optional<v4l2::Frame>& captured = dequeued.frame;
  pending_.store(false, std::memory_order_release);
  // The reader waits on the interlock rather than polling it, so lowering it is only half of the
  // handshake. Unconditional, because the reader is equally blocked whether or not a buffer was
  // behind the notification.
  pending_.notify_one();
  if (!captured) {
    if (dequeued.failed) {
      // Kept apart from the empty queue below, which a spurious wake makes routine. Without this
      // the two are one silence, and a driver that has stopped delivering looks exactly like a
      // device that simply had nothing ready. Throttled because a refusal repeats once per
      // activation for as long as it lasts.
      ++dequeue_failures_;
      if (dequeue_failures_ % kDeviceFailureComplaintInterval == 1U) {
        log_dequeue_failure(device_handle_->last_error(), dequeue_failures_);
      }
      // Returned rather than absorbed, and no tolerance applies here as it does to the wait. A
      // refused DQBUF is not a device with nothing to say -- the reader saw the descriptor become
      // readable, so a buffer was ready and the driver then declined to hand it over, or handed
      // over one it never mapped. Reporting that as a successful activation is what would let a
      // session whose device is broken finish and be recorded as having captured cleanly. The code
      // travels alone for the same reason it does at the requeue below: this runs per frame, and
      // the operation and errno are in the log above.
      return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
    }
    // A notification without a buffer behind it is possible and harmless; the next completed frame
    // posts again.
    return {};
  }
  // Cleared on the first frame that arrives, so a refusal the driver recovers from is reported
  // again if it returns rather than being throttled against a stale count.
  dequeue_failures_ = 0U;

  // Read before observing, because observing overwrites it and a regression is only diagnosable
  // against the value it moved back from.
  const std::uint64_t previous_sequence = sequence_.previous();
  const auto observation = sequence_.observe(captured->sequence);
  const CapturedPixels pixels{.data = captured->data, .bytes = captured->bytes};
  // A driver reports bytesused, not buffer capacity, so a short frame is an ordinary event rather
  // than a malfunction. Clamping the copy below is required for memory safety, but clamping alone
  // would publish a tensor whose declared extent exceeds the pixels behind it: the shape still says
  // every row is present. A consumer that is told nothing cannot distinguish fabricated rows from a
  // genuinely dark scene.
  const bool truncated = pixels.bytes < capture_bytes_;

  holoscan::EmitOptions options{};
  options.frame_id = captured->sequence;
  // Two independent conditions, because a publishable capture time needs both and V4L2 states them
  // in separate flag fields. The domain has to be one this host can project from, and the source
  // has to be the instant the SDK's capture time actually means -- the start of integration.
  //
  // V4L2 defaults its source to end-of-frame, which trails the start of integration by the exposure
  // plus the readout. Projecting that and publishing it would be well-typed and wrong by a margin
  // that grows with exposure, and wrong in the direction nothing downstream can detect: every
  // consumer would receive a plausible number. Absent is worse for the consumer that wanted a
  // timestamp and better for every consumer that would have aligned, fused, or measured latency
  // against a false one, and Core never substitutes arrival or publication time for the gap.
  const bool projectable_domain =
      device_handle_->timestamp_domain() == v4l2::TimestampDomain::kMonotonic;
  const bool integration_start =
      captured->timestamp_source == v4l2::TimestampSource::kStartOfExposure;
  if (projectable_domain && integration_start) {
    if (const auto projected = clock_.project(captured->timestamp_ns)) {
      options.capture_time = *projected;
    }
  }
  // Complained about only when the timestamp was otherwise usable. A device that names no clock is
  // already unprojectable and says so elsewhere; repeating it here as a source problem would name
  // the wrong cause.
  if (projectable_domain && !integration_start) {
    ++declined_timestamps_;
    if (declined_timestamps_ % kDeclinedTimestampComplaintInterval == 1U) {
      log_declined_timestamp(captured->timestamp_source, declined_timestamps_);
    }
  } else {
    declined_timestamps_ = 0U;
  }
  if (observation.gap > 0U) {
    // The frames the driver produced and we never collected are already gone. Marking the next
    // frame we do publish is the only way the loss reaches anyone.
    options.flags |= holoscan::SampleFlags::kDegraded;
  }
  if (observation.regressed) {
    // kStart resets the tracker alongside stream_on precisely so a restart this operator performed
    // does not land here, which is what makes a regression that does land meaningful: the driver
    // restarted its own counter, or the counter wrapped at a width narrower than the tracker reads.
    // Either way the gap arithmetic cannot run backwards, so an unknown number of samples may sit
    // across the discontinuity and the frame carries the same flag a counted gap would.
    //
    // A driver that never fills the counter regresses on every frame and so degrades every frame.
    // That is the honest answer rather than an oversight: continuity cannot be vouched for when the
    // only evidence for it is absent. It is also why the complaint is throttled.
    ++sequence_regressions_;
    if (sequence_regressions_ % kSequenceRegressionComplaintInterval == 1U) {
      log_sequence_regression(observation.sequence, previous_sequence, sequence_regressions_);
    }
    options.flags |= holoscan::SampleFlags::kDegraded;
  } else {
    // Cleared on the first counter that advances, so a driver that recovers is reported again if it
    // regresses later rather than being throttled against a stale count.
    sequence_regressions_ = 0U;
  }
  if (captured->driver_error) {
    // The driver delivered the buffer and told us it is corrupt. Publishing it unmarked would let a
    // consumer treat known-bad pixels as measurements.
    options.flags |= holoscan::SampleFlags::kInvalid;
  }
  if (truncated) {
    // Degraded rather than invalid, because the rows that did arrive are real measurements and a
    // consumer that can use a partial frame should be allowed to.
    options.flags |= holoscan::SampleFlags::kDegraded;
  }

  const holoscan::expected<void, holoscan::Error> published =
      publish_frame(frame, pixels, width_, height_, frame_id_, options, context.cuda_stream());
  // The buffer must go back to the driver whether or not publication succeeded; a buffer that is
  // never requeued starves the capture pool a few frames later rather than here.
  if (!device_handle_->requeue(captured->index)) {
    // Both failures are reported here, because only one of them can be returned. This is the only
    // Device call outside a lifecycle stage, so device_failure() does not fit -- it answers with a
    // LifecycleStatus -- and without a replacement the operation and errno the device already
    // recorded are dropped, taking any publication error behind them along too.
    log_failure("requeue", device_handle_->last_error());
    if (!published) {
      log_publication_failure(published.error());
    }
    // The requeue is what propagates. A buffer the driver never takes back is gone from the capture
    // pool for good and starves the device a few frames later, while a failed publication costs the
    // one frame it was carrying. The code travels alone: this runs per frame, and what a message
    // would have carried is in the log above.
    return holoscan::make_unexpected(holoscan::Error{holoscan::ErrorCode::kFailure});
  }
  return published;
}

}  // namespace holoscan::holoscan_camera
