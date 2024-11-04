// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <fstream>
#include <chrono>
#include <cstdint>
#include <matx.h>
#include "holoscan/holoscan.hpp"
#include "vrt_time.hpp"

using namespace matx;

using complex = cuda::std::complex<float>;

namespace holoscan::ops {
class FileReader : public Operator {
 public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(FileReader)

    void initialize() override;
    void setup(OperatorSpec& spec) override;
    void compute(InputContext& input, OutputContext& output, ExecutionContext& context) override;

 private:
    tensor_t<complex, 2> rf_data;
    std::ifstream file;
    Parameter<std::string> file_name;
    Parameter<int> burst_size;
    Parameter<uint32_t> stream_id;
    Parameter<double> rf_ref_freq_hz;
    Parameter<double> bandwidth_hz;
    Parameter<double> sample_rate_sps;
    Parameter<float> ref_level_dbm;
    Parameter<float> gain_db;

    int num_chunks;
    int chunk_index = 0;
    VrtTime vrt_time;
};

}  // namespace holoscan::ops
