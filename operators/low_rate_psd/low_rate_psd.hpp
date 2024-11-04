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
  void reset();
  tensor_t<float, 1> psdAccumulator;
  tensor_t<int8_t, 1> psdAverage;
  Parameter<int> burst_size;
  Parameter<uint32_t> num_averages;
  int fft_count;
  holoscan::MetadataDictionary current_meta;
};

}  // namespace holoscan::ops
