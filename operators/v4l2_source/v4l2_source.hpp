// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>
#include <holoscan/time/time_point.hpp>

struct v4l2_buffer;

namespace holoscan::examples::v4l2_depth {

/**
 * @brief EA1 clock-polled V4L2 source producing CUDA-device YUYV tensors.
 *
 * The operator negotiates one exact, single-plane YUYV capture mode and uses
 * V4L2 MMAP buffers. Each dequeued frame is copied row-by-row to a plan-owned
 * CUDA tensor with shape `[height, width, 2]`. The copy is completed before the
 * kernel buffer is returned to the camera driver.
 *
 * This is an early implementation for demo purposes only.
 * An alternate implementation with full support is planned for a later release.
 */
class V4L2SourceOp final : public holoscan::Operator<> {
 public:
  /**
   * @param device V4L2 device path, for example `/dev/video0`.
   * @param width Required capture width in pixels.
   * @param height Required capture height in pixels.
   * @param fps Required capture rate.
   */
  V4L2SourceOp(std::string device, int width, int height, int fps = 30);
  ~V4L2SourceOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  /// CUDA-device uint8 tensor with logical shape `[height, width, 2]`.
  holoscan::Output<holoscan::Tensor> frame;

 private:
  struct MappedBuffer {
    void* address{};
    std::size_t length{};
  };

  [[nodiscard]] int xioctl(unsigned long request,  // NOLINT(runtime/int)
                           void* argument) const noexcept;
  void cleanup() noexcept;
  [[nodiscard]] holoscan::TimePoint capture_time(
      const v4l2_buffer& buffer, holoscan::TimePoint activation_time) const noexcept;

  std::string device_;
  int width_{};
  int height_{};
  int fps_{};
  std::chrono::nanoseconds poll_period_{};
  std::size_t row_bytes_{};
  std::size_t bytes_per_line_{};
  std::size_t frame_bytes_{};
  int fd_{-1};
  bool streaming_{};
  std::vector<MappedBuffer> buffers_;
  std::uint64_t frame_id_{};
};

}  // namespace holoscan::examples::v4l2_depth
