// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

/// @file
/// @brief A V4L2 capture source written the way an SDK 5.x sensor source is meant to be written.
///
/// The operator itself is small. What makes it a reference is the set of obligations it discharges,
/// none of which are specific to V4L2, and all of which a capture source has whether or not it
/// bothers to meet them:
///
/// - It publishes a typed payload (`holoscan::schema::ImageT`) rather than a bare tensor, so a
///   consumer learns the geometry, the encoding, the coordinate frame, and the acquisition instant
///   from the sample instead of from documentation.
/// - It is woken by the device rather than by a clock. A `holoscan::OnClock` trigger on a capture
///   source is a guess about when frames exist; a reader thread posting one notification per
///   completed buffer is not a guess.
/// - It spreads the V4L2 streaming sequence across the lifecycle stages that own each step, so a
///   restart that keeps its buffers does not re-open the descriptor.
/// - It projects the driver's `CLOCK_MONOTONIC` buffer timestamp into the graph's clock domain, and
///   declines to project at all when the driver did not name its clock.
/// - It reports what it lost. A sequence gap, a counter that goes backwards, and a short frame
///   each mark the published sample degraded, and a driver-marked corrupt frame marks it invalid,
///   so a consumer can tell a degraded frame from a clean one. The marks are per-sample; no
///   aggregate dropped-frame count is published.
///
/// Those five behaviours come from `holoscan::sensor_io`, which supplies the state machines
/// (\ref holoscan::sensor_io::StageGuards, \ref holoscan::sensor_io::ClockDiscipline,
/// \ref holoscan::sensor_io::SequenceTracker) rather than leaving each sensor to reimplement them.
/// A second sensor in this module should hold the same three members and declare the same stage
/// hooks; only the device calls between them differ.
///
/// There is no dependency on Holoscan Sensor Bridge here, directly or transitively: the capture
/// path is the kernel's V4L2 MMAP interface and nothing else.

// clang-format off: IncludeBlocks: Regroup would sort <holoscan/...> in among the standard library
// headers, which is the order cpplint's build/include_order check rejects.
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stop_token>  // NOLINT(build/include_order)
#include <string>
#include <string_view>
#include <thread>  // NOLINT(build/c++11)

#include <dlpack/dlpack.h>  // NOLINT(build/include_order)
#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
// Registers ImageT as a legal port payload. Without it, Output<schema::ImageT> below fails the
// PublishedPayload constraint with a message about an unsatisfied disjunction that names no header.
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/sensor_io/sensor_io.hpp>
// clang-format on

namespace holoscan::holoscan_camera {

namespace v4l2 {
class Device;
}  // namespace v4l2

/// @brief Element type of the published frame.
/// @note A packed YUYV byte, not a pixel. `ImageEncoding_YUYV` is authoritative for how the bytes
/// group into pixels, so the Tensor element stays a scalar byte and the grouping is not restated as
/// a lane count, which the schema rejects for every encoding that names its own organization.
inline constexpr DLDataType kFrameElementDtype{kDLUInt, 8U, 1U};

/// @brief Coordinate frame a camera image is expressed in unless the caller names another.
inline constexpr std::string_view kDefaultCameraFrameId = "camera_optical_frame";

/// @brief Capture YUYV frames from a V4L2 device and publish them as typed, stamped images.
///
/// @note The published encoding is `ImageEncoding_YUYV`: the packed 4:2:2 frame exactly as the
/// driver delivered it, chroma included. The schema names YUYV directly, so there is no longer a
/// choice between publishing only the plane the schema could describe and declaring
/// `ImageEncoding_CUSTOM` to carry every byte while telling the consumer nothing about how to read
/// it. Discarding chroma was a cost of the older vocabulary rather than a property of the device,
/// and a capture source that drops half the measurement it was given cannot be a reference for one
/// that must not.
///
/// @note The Tensor is rank 2 and one byte per element, shaped `[height, 2 * width]`, so a row is
/// its own byte extent. The pixel geometry is not inferable from that shape and is not meant to be:
/// `width`, `height`, and `encoding` on the descriptor are where a consumer reads it. Packing the
/// pixel structure into the shape instead would state the same organization the encoding already
/// fixes, and the two could then disagree.
///
/// @note Where the frame lands is a constructor choice, because the placement is frozen into the
/// port's Tensor contract at `setup()` and cannot be renegotiated once the plan is compiled. The
/// driver always delivers into a pageable mmap buffer, so every placement copies once out of it;
/// the choice is what the copy targets. `kHost` keeps the frame in ordinary pageable memory, which
/// costs nothing here but makes any later device transfer bounce through a driver staging buffer.
/// `kPinnedHost` costs the same copy and leaves the frame host-readable while making that later
/// transfer a direct DMA. `kCudaDevice` moves the frame to the GPU here instead of downstream,
/// which suits a graph whose consumers are all device-resident and wastes a round trip for one
/// whose consumers are not. Placement between two operators that disagree is an edge concern --
/// see `ConnectionOptions::memory_conversion` -- so prefer this parameter for stating where the
/// producer's own consumers want the data, not for repairing a mismatch.
///
/// @warning `kCudaDevice` additionally requires the *application* to name a GPU, because
/// compilation rejects a device-resident pool with no device rather than defaulting to ordinal
/// zero. Bind it alongside the graph:
/// @code
/// holoscan::CompileOptions options{};
/// options.deployment.bind_tensor_output_device(holoscan::TensorOutputDevicePlacement{
///     .operator_path = "camera", .output_port = "frame", .device = holoscan::DeviceId{0}});
/// const holoscan::ExecutionPlan plan = holoscan::compile(graph, std::move(options));
/// @endcode
/// Without it the plan is rejected with `DEVICE_UNRESOLVED`. The binding must be absent for
/// `kHost` and process-local `kPinnedHost`, where it is instead rejected as
/// `TENSOR_DEVICE_BINDING_UNEXPECTED`, so it cannot simply be set unconditionally.
class V4l2CaptureOp final : public holoscan::Operator<> {
 public:
  /// @brief Construct a capture source for one device node and one capture mode.
  /// @param device Device node path, typically `/dev/video0`.
  /// @param width Capture width in pixels; must be even for YUYV.
  /// @param height Capture height in pixels.
  /// @param fps Requested frames per second.
  /// @param frame_id Coordinate frame the images are expressed in, carried in `Header::frame_id`.
  /// @param memory_kind Where the published frame is allocated. `kHost`, `kPinnedHost`, and
  ///                    `kCudaDevice` are accepted; see the class note for how to choose. Defaults
  ///                    to `kHost` so that adding this parameter does not silently change the
  ///                    memory profile of a graph that never asked for one.
  /// @throws std::invalid_argument if the requested mode cannot describe a YUYV frame, or if
  ///         \p memory_kind is not one of the three accepted placements.
  explicit V4l2CaptureOp(std::string device = "/dev/video0", int width = 640, int height = 480,
                         int fps = 30, std::string frame_id = std::string{kDefaultCameraFrameId},
                         holoscan::MemoryKind memory_kind = holoscan::MemoryKind::kHost);
  ~V4l2CaptureOp() override;

  V4l2CaptureOp(const V4l2CaptureOp&) = delete;
  V4l2CaptureOp& operator=(const V4l2CaptureOp&) = delete;
  V4l2CaptureOp(V4l2CaptureOp&&) = delete;
  V4l2CaptureOp& operator=(V4l2CaptureOp&&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;

  // Lifecycle stage hooks. Each body runs only if the stage before it left something to stand on,
  // which is what the guard decides; a body whose precondition is unmet is never invoked and
  // reports kSkipped. A restart re-entering at kConfigure starts from a clean slate.
  holoscan::LifecycleStatus on_configure(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_discover(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_allocate(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_arm(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_stop(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_release(holoscan::LifecycleContext&) noexcept;

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext&) override;

  [[nodiscard]] const std::string& device() const noexcept { return device_; }
  [[nodiscard]] const std::string& frame_id() const noexcept { return frame_id_; }
  [[nodiscard]] int width() const noexcept { return width_; }
  [[nodiscard]] int height() const noexcept { return height_; }
  [[nodiscard]] int fps() const noexcept { return fps_; }

  /// @brief Placement the published frame is allocated in.
  /// @note This is the value declared to the port at `setup()`. The write path branches on the
  /// placement the compiled plan actually froze rather than on this, so the two cannot drift.
  [[nodiscard]] holoscan::MemoryKind memory_kind() const noexcept { return memory_kind_; }

  /// @brief Bytes in one packed YUYV frame, which the driver delivers and the operator publishes.
  /// @note One figure rather than a captured size and a published size, because the two are now the
  /// same act: the frame is forwarded whole. It is also what `ImageEncoding_YUYV` independently
  /// requires of a `width` by `height` frame. An even width is what makes the schema's row of
  /// `4 * ceil(width / 2)` bytes equal the `2 * width` the driver delivers, and the constructor
  /// rejects an odd width, which is what keeps those two derivations from parting.
  [[nodiscard]] std::size_t capture_bytes() const noexcept { return capture_bytes_; }

  holoscan::Output<holoscan::schema::ImageT> frame;

 private:
  [[nodiscard]] holoscan::LifecycleStatus device_failure(std::string_view stage) noexcept;

  /// @brief Release everything the current epoch's forward stages acquired.
  /// @return Always \ref holoscan::LifecycleStatus::kOk.
  /// @note Factored out of \ref on_release because it runs from two places: as the guarded
  /// `kRelease` body, and directly when the guard withholds that body because `kAllocate` never
  /// claimed ownership while `kConfigure` had already opened the descriptor. Every line of it is
  /// idempotent so the second path cannot double-release.
  [[nodiscard]] holoscan::LifecycleStatus release_resources() noexcept;

  void ensure_clock_bound(holoscan::ExecutionContext&) noexcept;
  void reader_loop(const std::stop_token& stop_token) noexcept;
  [[nodiscard]] bool post_until_accepted(const std::stop_token& stop_token) noexcept;

  std::string device_;
  std::string frame_id_;
  int width_{};
  int height_{};
  int fps_{};
  holoscan::MemoryKind memory_kind_{holoscan::MemoryKind::kHost};
  std::size_t capture_bytes_{};

  std::unique_ptr<v4l2::Device> device_handle_;
  holoscan::NotificationSource frame_ready_;
  holoscan::NotificationSender sender_;
  std::jthread reader_;
  std::atomic<bool> pending_{};

  /// @brief The failure the reader gave up on, or zero when it has not given up.
  /// @note The reader thread holds no activation, so it cannot fail a run by returning. This is the
  /// one-way channel that lets it hand the cause to the next activation, which can. Cleared by the
  /// activation that reports it, so a device lost once is not reported by every later activation.
  std::atomic<std::uint64_t> reader_failure_{};

  /// @brief Consecutive attempts to settle the clock discipline that have failed.
  /// @note Zero whenever the discipline is settled. Priming is reattempted from every activation
  /// until it succeeds, so this is what bounds how often a device that never brackets cleanly
  /// reports it.
  std::uint32_t clock_prime_failures_{};

  /// @brief Consecutive activations whose dequeue the driver refused.
  /// @note Zero once a frame arrives. A refusal repeats for as long as the condition lasts, so this
  /// is what bounds how often it is reported.
  std::uint32_t dequeue_failures_{};

  /// @brief Consecutive frames whose device counter moved backwards or repeated.
  /// @note Zero once a counter advances again. A driver that never fills the counter regresses on
  /// every frame, so this is what bounds how often that is reported.
  std::uint32_t sequence_regressions_{};

  /// @brief Consecutive frames whose timestamp was projectable but did not mark integration start.
  /// @note Zero once a frame arrives that either carries a publishable capture time or has no
  /// projectable clock at all. Most drivers stamp end-of-frame for every frame they ever deliver,
  /// so this is what keeps a fixed device property from logging once per frame forever.
  std::uint32_t declined_timestamps_{};

  holoscan::sensor_io::StageGuards guards_;
  holoscan::sensor_io::ClockDiscipline clock_;
  holoscan::sensor_io::SequenceTracker sequence_;
};

}  // namespace holoscan::holoscan_camera
