// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <X11/Xlib.h>
#include <X11/Xutil.h>

#include <cstddef>
#include <string>

#include <holoscan/core/domain/tensor.hpp>
#include <holoscan/core/errors.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/expected.hpp>
#include <holoscan/core/operator.hpp>
#include <holoscan/core/operator_spec.hpp>
#include <holoscan/core/port.hpp>
#include <holoscan/core/temporal_contract.hpp>

namespace holoscan::examples::v4l2_depth {

/**
 * @brief Display CUDA-device BGRA tensors in an X11 window.
 *
 * The sink explicitly copies each contiguous `[height, width, 4]` tensor to
 * XImage-owned host storage on the runtime CUDA stream. If no X server is
 * available, the operator remains active and drains frames without rendering.
 */
class X11DisplayOp final : public holoscan::Operator<> {
 public:
  X11DisplayOp(int width, int height, std::string title = "Holoscan V4L2 Depth");
  ~X11DisplayOp() override;

  void setup(holoscan::OperatorSpec& spec) override;
  [[nodiscard]] holoscan::Contract contract() const override;
  void start() override;
  void stop() override;
  [[nodiscard]] holoscan::expected<void, holoscan::Error> compute(
      holoscan::ExecutionContext& context) override;

  holoscan::Input<holoscan::Tensor> input;

 private:
  void cleanup() noexcept;
  void disable(const char* reason) noexcept;
  void process_events();

  int width_{};
  int height_{};
  std::string title_;
  std::size_t frame_bytes_{};
  bool disabled_{};
  Display* display_{};
  Window window_{};
  GC gc_{};
  XImage* image_{};
  Atom wm_delete_window_{};
};

}  // namespace holoscan::examples::v4l2_depth
