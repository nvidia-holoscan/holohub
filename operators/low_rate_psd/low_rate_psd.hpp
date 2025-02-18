// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <matx.h>
#include "holoscan/holoscan.hpp"

using namespace matx;

namespace holoscan::ops {
class LowRatePSD : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(LowRatePSD)

  LowRatePSD() = default;

  void initialize() override;
  void setup(OperatorSpec& spec) override;
  void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
  tensor_t<int8_t, 2> outputs;
  tensor_t<double, 1> maxima;
  tensor_t<double, 1> minima;
  Parameter<int> burst_size;
  Parameter<int> num_bursts;
  Parameter<uint16_t> num_channels;
  Parameter<uint32_t> num_averages;
};

}  // namespace holoscan::ops
