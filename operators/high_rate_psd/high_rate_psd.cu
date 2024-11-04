// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "high_rate_psd.hpp"

namespace holoscan::ops {

void HighRatePSD::setup(OperatorSpec& spec) {
    spec.input<tensor_t<complex, 1>>("in");
    spec.output<tensor_t<float, 1>>("out");
    spec.param(burst_size,
               "burst_size",
               "Burst size"
               "Number of samples to process in each burst");
}

void HighRatePSD::initialize() {
    holoscan::Operator::initialize();
    make_tensor(psdOut, {burst_size.get()}, MATX_DEVICE_MEMORY);
    scale_factor = 1.0 / pow(burst_size.get(), 2);
}

void HighRatePSD::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    auto fft_data = op_input.receive<tensor_t<complex, 1>>("in").value();
    (psdOut = abs2(fft_data) * scale_factor).run();
    op_output.emit(psdOut, "out");
}
}  // namespace holoscan::ops
