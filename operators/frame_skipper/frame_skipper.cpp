// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "frame_skipper/frame_skipper.hpp"

#include <stdexcept>
#include <utility>

#include <holoscan/core/payload_options.hpp>

#include "v4l2_depth_common/tensor_utils.hpp"

namespace holoscan::examples::v4l2_depth {

FrameSkipperOp::FrameSkipperOp(std::uint32_t keep_one_in_n)
    : keep_one_in_n_(keep_one_in_n) {
  if (keep_one_in_n_ == 0U) {
    throw std::invalid_argument("FrameSkipperOp keep_one_in_n must be positive");
  }
}

void FrameSkipperOp::setup(holoscan::OperatorSpec& spec) {
  constexpr holoscan::TensorRepresentation kDeviceByteImage{
      .memory_kind = holoscan::MemoryKind::kCudaDevice,
      .dtype = kUInt8Dtype,
      .rank = 3U,
  };
  // Retaining queued frames would only make the inference branch older. The
  // graph's latest-value connection provides the intended bounded decimation.
  spec.input(input, "input")
      .queue_depth(1U)
      .expects_tensor(holoscan::TensorInputSpec{.representation = kDeviceByteImage});
  // This is a pass-through publication declaration, not a direct tensor
  // allocation authority. The retained input sample owns the backing storage.
  spec.output(output, "output")
      .max_emits_per_compute(1U)
      .produces_tensor(holoscan::TensorOutputSpec{.representation = kDeviceByteImage});
}

holoscan::Contract FrameSkipperOp::contract() const {
  holoscan::Contract result;
  result.trigger(holoscan::OnEach{input});
  return result;
}

holoscan::expected<void, holoscan::Error> FrameSkipperOp::compute(
    holoscan::ExecutionContext&) {
  auto sample = input.receive();
  if (!sample) {
    return holoscan::make_unexpected(std::move(sample).error());
  }

  ++received_;
  if ((received_ % keep_one_in_n_) != 0U) {
    return {};
  }
  return output.emit(std::move(*sample));
}

}  // namespace holoscan::examples::v4l2_depth
