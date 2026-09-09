// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <chrono>
#include <vector>

#include <holoscan/holoscan.hpp>

#include <matx.h>

using complex = cuda::std::complex<float>;

namespace holoscan::ops {

class LogOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(LogOp)

  using in_t = std::tuple<matx::tensor_t<complex, 2>, cudaStream_t>;

  LogOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output,
      ExecutionContext& context) override;

 private:
  Parameter<int> num_channels_;
  Parameter<int> log_interval_;
  Parameter<bool> log_data_;
  std::vector<int64_t> total_samples_;
  std::vector<std::chrono::steady_clock::time_point> start_;
  std::vector<std::chrono::steady_clock::duration> elapsed_;
};

}  // namespace holoscan::ops
