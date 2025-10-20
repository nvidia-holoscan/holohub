// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cmath>
#include <matx.h>
#include "holoscan/holoscan.hpp"

using namespace matx;

using complex = cuda::std::complex<float>;

namespace holoscan::ops {
class HighRatePSD : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(HighRatePSD)

  HighRatePSD() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  tensor_t<float, 3> outputs;
  Parameter<uint16_t> num_channels;
  Parameter<int> burst_size;
  Parameter<int> num_bursts;
  double scale_factor;
};

}  // namespace holoscan::ops
