// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Private backend boundary. No Jetson headers enter the installed operator interface.
// clang-format off
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <holoscan/core/tensor_output_loan.hpp>

#include "argus_capture_op/argus_capture_op.hpp"
// clang-format on

namespace holoscan::holoscan_camera::argus {

struct CapturedFrame {
  std::size_t slot{};
  ArgusFrameMetadataT metadata;
};

/// acquire() returns exclusive buffer ownership until release(). An empty result means timeout;
/// all session/stream failures throw with the failing operation and native status.
class CaptureService {
 public:
  virtual ~CaptureService() = default;
  virtual void start() = 0;
  virtual void stop() = 0;
  virtual std::optional<CapturedFrame> acquire(std::uint32_t timeout_ms) = 0;
  virtual void release(std::size_t slot) = 0;
  /// Copies on the write guard's producer stream and completes before returning, including on GPU.
  virtual void copy_frame(std::size_t slot, const holoscan::TensorOutputWriteGuard& writer,
                          holoscan::MemoryKind placement) = 0;
  virtual std::optional<std::int64_t> timestamp_now_ns() noexcept = 0;
};

std::unique_ptr<CaptureService> make_service(const ArgusCaptureConfig& config);
std::size_t validate_config(const ArgusCaptureConfig& config);

/// Fixed-capacity queue. Caller serializes access and releases any returned rejected frame.
/// Two native slots remain outside this queue: one for compute and one for capture progress.
class FrameQueue {
 public:
  explicit FrameQueue(std::size_t capacity) : slots_(capacity) {}
  std::optional<CapturedFrame> push(CapturedFrame frame, ArgusDropPolicy policy) {
    std::optional<CapturedFrame> dropped;
    if (size_ == slots_.size()) {
      if (policy == ArgusDropPolicy::kDropNewest) {
        return frame;
      }
      dropped = pop();
    }
    slots_[(head_ + size_) % slots_.size()] = std::move(frame);
    ++size_;
    return dropped;
  }
  std::optional<CapturedFrame> pop() {
    if (size_ == 0) return std::nullopt;
    auto frame = std::move(slots_[head_]);
    slots_[head_].reset();
    head_ = (head_ + 1) % slots_.size();
    --size_;
    return frame;
  }
  bool empty() const noexcept { return size_ == 0; }
  std::size_t size() const noexcept { return size_; }

 private:
  std::vector<std::optional<CapturedFrame>> slots_;
  std::size_t head_{};
  std::size_t size_{};
};

}  // namespace holoscan::holoscan_camera::argus
