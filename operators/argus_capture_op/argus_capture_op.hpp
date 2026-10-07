// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// clang-format off
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/sensor_io/image_metadata.hpp>
#include <holoscan/sensor_io/sensor_schema_package.hpp>

#include "argus_capture_op/argus_frame_metadata_schema_traits.hpp"
// clang-format on

namespace holoscan::holoscan_camera {

enum class ArgusPixelFormat : std::uint8_t { kNv12, kI420 };
enum class ArgusDropPolicy : std::uint8_t { kDropOldest, kDropNewest };

/// Configuration is fixed before setup(), including the tensor's geometry and placement.
/// Camera and mode indices are those reported by libargus, not /dev/videoN indices.
struct ArgusCaptureConfig {
  std::uint32_t camera_index{0};
  std::uint32_t sensor_mode{0};
  std::uint32_t width{1920};
  std::uint32_t height{1080};
  double fps{30.0};
  ArgusPixelFormat pixel_format{ArgusPixelFormat::kNv12};
  std::uint32_t buffer_count{6};
  ArgusDropPolicy drop_policy{ArgusDropPolicy::kDropOldest};
  std::uint32_t timeout_ms{2000};
  std::int32_t cuda_device{0};
  std::string frame_id{"camera_optical_frame"};
  /// Normal capture-to-GPU operation uses kCudaDevice. Host placements copy explicitly.
  holoscan::MemoryKind memory_kind{holoscan::MemoryKind::kCudaDevice};
  /// Optional, camera-calibrated correction: integration_start = VI_SOF + offset.
  /// Leave unset unless the sensor's timing establishes this relationship. Argus's generic
  /// first-data timestamp is always carried in metadata, but is not an integration timestamp.
  std::optional<std::int64_t> integration_start_offset_ns;
};

/// Jetson libargus source with bounded native buffers and descriptor-backed image output.
///
/// Owns one capture session. The worker posts OnNotified readiness; compute() copies the acquired
/// image into an HSDK pool and returns the native buffer before publishing. GPU placement requires
/// bind_tensor_output_device({.operator_path="camera", .output_port="frame", .device=DeviceId{0}}).
/// frame and metadata carry the same device sequence; publication on two ports is not atomic.
class ArgusCaptureOp final : public holoscan::Operator<> {
 public:
  explicit ArgusCaptureOp(ArgusCaptureConfig config = {});
  ~ArgusCaptureOp() override;
  ArgusCaptureOp(const ArgusCaptureOp&) = delete;
  ArgusCaptureOp& operator=(const ArgusCaptureOp&) = delete;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::LifecycleStatus on_configure(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_allocate(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_arm(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_start(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_stop(holoscan::LifecycleContext&) noexcept;
  holoscan::LifecycleStatus on_release(holoscan::LifecycleContext&) noexcept;

  [[nodiscard]] const ArgusCaptureConfig& config() const noexcept { return config_; }
  [[nodiscard]] std::size_t frame_bytes() const noexcept { return frame_bytes_; }

  holoscan::Output<holoscan::schema::ImageT> frame;
  holoscan::Output<ArgusFrameMetadataT> metadata;

 private:
  struct Impl;
  ArgusCaptureConfig config_;
  std::size_t frame_bytes_;
  std::unique_ptr<Impl> impl_;
  holoscan::NotificationSource frame_ready_;
};

}  // namespace holoscan::holoscan_camera
