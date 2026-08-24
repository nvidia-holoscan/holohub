// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "holoscan/holoscan.hpp"
#include <matx.h>

using complex = cuda::std::complex<float>;

namespace holoscan::ops {
class SpectrumMagnitudeOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SpectrumMagnitudeOp)

  SpectrumMagnitudeOp() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  matx::tensor_t<float, 2> outputs_;
  matx::tensor_t<float, 3> abs2_buf_;
  Parameter<int> burst_size_;
  Parameter<int> num_bursts_;
  Parameter<uint16_t> num_channels_;
  Parameter<uint32_t> num_averages_;
  float scale_factor_;
};

}  // namespace holoscan::ops
