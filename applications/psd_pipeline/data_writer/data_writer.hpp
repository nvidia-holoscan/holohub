// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <fstream>
#include <matx.h>
#include "holoscan/holoscan.hpp"

using namespace matx;
using complex = cuda::std::complex<float>;

namespace holoscan::ops {
class DataWriter : public Operator {
 public:
     HOLOSCAN_OPERATOR_FORWARD_ARGS(DataWriter)

     DataWriter() = default;

     void setup(OperatorSpec& spec) override;
     void initialize() override;
     void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;

 private:
     tensor_t<complex, 2> data_host;
     Parameter<int> burst_size;
     Parameter<int> num_bursts;
};

}  // namespace holoscan::ops
