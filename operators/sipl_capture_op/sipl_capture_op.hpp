// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>

#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/lifecycle.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/readiness_source.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/core/tensor_output_loan.hpp>
#include <holoscan/schema/flatbuffer_schema_package.hpp>
#include <holoscan/sensor_io/clock_discipline.hpp>

// Generated at build time by holoscan_add_flatbuffer_schema (sipl_frame_metadata.fbs).
// Provides holoscan::holoscan_camera::SIPLFrameMetadataT and schema_identity<> specialization.
#include "sipl_capture_op/sipl_frame_metadata_schema_traits.hpp"
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>
#include <holoscan/sensor_io/lifecycle_guards.hpp>
#include <holoscan/sensor_io/sequence_tracker.hpp>

namespace holoscan::holoscan_camera {

class SIPLCaptureService;

/**
 * @brief Holoscan SDK 5.x SIPL camera capture operator.
 *
 * Acquires one frame per compute() activation from a single camera managed by
 * SIPLCaptureService and emits it as a holoscan::Tensor on the `frame` output port.
 *
 * Multiple SIPLCaptureOp instances can share one SIPLCaptureService when
 * capturing from a multi-camera rig; the service starts streaming when the
 * first operator arms and tears down when the last one stops.
 *
 * No dependency on HSB / hololink headers. SIPL internals (which use HSB
 * through the UDDF drivers) are statically linked inside the driver stack and
 * do not surface here.
 *
 * Usage:
 * @code
 *   auto service = std::make_shared<SIPLCaptureService>(...);
 *   auto camera = graph.op<SIPLCaptureOp>("camera", service, 0U);  // camera_index 0
 * @endcode
 *
 * The service and camera index are required construction inputs captured by the graph.
 *
 * @note Where the frame lands is a construction choice because the placement is frozen into the
 * port's Tensor contract at `setup()`.
 */
class SIPLCaptureOp final : public holoscan::Operator<> {
 public:
  /// @brief Construct a capture source backed by a shared SIPL camera rig.
  /// @param service Shared SIPL capture service managing the camera rig. Must not be null.
  /// @param camera_index Zero-based index of the camera within the rig. Validated during discover.
  /// @param memory_kind Where the published frame is allocated. `kHost`, `kPinnedHost`, and
  ///                    `kCudaDevice` are accepted; see the class note for how to choose. Defaults
  ///                    to `kCudaDevice` so that a caller which never names a placement keeps this
  ///                    operator's original memory profile. `kCudaDevice` additionally requires the
  ///                    application to bind a device via
  ///                    `CompileOptions::deployment.bind_tensor_output_device()`; the binding must
  ///                    be absent for the two host kinds.
  /// @throws std::invalid_argument if \p service is null or \p memory_kind is unsupported.
  explicit SIPLCaptureOp(
      std::shared_ptr<SIPLCaptureService> service,
      std::uint32_t camera_index,
      holoscan::MemoryKind memory_kind = holoscan::MemoryKind::kCudaDevice);

  // Declares the two output ports (frame, sensor_data), the frame-ready
  // NotificationSource, and registers the sensor lifecycle hooks.
  void setup(holoscan::OperatorSpec& spec) override;

  // Returns an OnNotified contract so the scheduler activates compute() only when
  // the acquire thread signals that a new frame is available.
  [[nodiscard]] holoscan::Contract contract() const override;

  // Lifecycle stage hooks. Each body runs only if the stage before it left something to stand on,
  // which is what the guard decides; a body whose precondition is unmet is never invoked and
  // reports kSkipped. A restart re-entering at kConfigure starts from a clean slate.
  [[nodiscard]] LifecycleStatus on_configure(LifecycleContext& ctx) noexcept;
  [[nodiscard]] LifecycleStatus on_discover(LifecycleContext& ctx) noexcept;
  [[nodiscard]] LifecycleStatus on_allocate(LifecycleContext& ctx) noexcept;
  [[nodiscard]] LifecycleStatus on_arm(LifecycleContext& ctx) noexcept;
  [[nodiscard]] LifecycleStatus on_start(LifecycleContext& ctx) noexcept;
  [[nodiscard]] LifecycleStatus on_stop(LifecycleContext& ctx) noexcept;
  [[nodiscard]] LifecycleStatus on_release(LifecycleContext& ctx) noexcept;

  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  // Output ports
  holoscan::Output<holoscan::schema::ImageT> frame;
  holoscan::Output<SIPLFrameMetadataT> sensor_data;

  /// @brief Placement the published frame is allocated in.
  [[nodiscard]] holoscan::MemoryKind memory_kind() const noexcept { return memory_kind_; }

 private:
  void ensure_clock_settled(holoscan::ExecutionContext& context) noexcept;

  std::shared_ptr<SIPLCaptureService> service_;
  std::uint32_t camera_index_;
  holoscan::MemoryKind memory_kind_{ holoscan::MemoryKind::kCudaDevice };
  std::uint32_t timeout_us_{ 1'000'000 };

  holoscan::sensor_io::StageGuards guards_;
  holoscan::sensor_io::SequenceTracker sequence_tracker_;

  std::uint64_t frame_count_{ 0 };
  std::uint32_t embedded_top_lines_{ 0 };
  std::uint32_t embedded_bottom_lines_{ 0 };

  holoscan::sensor_io::ClockDiscipline clock_discipline_;
  holoscan::NotificationSource frame_ready_;
  holoscan::NotificationSender sender_;
};

}  // namespace holoscan::holoscan_camera
