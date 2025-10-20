// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "high_rate_psd.hpp"

using in_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;
using out_t = std::tuple<tensor_t<float, 2>, cudaStream_t>;

namespace holoscan::ops {

void HighRatePSD::setup(OperatorSpec& spec) {
    spec.input<in_t>("in");
    spec.output<out_t>("out");
    spec.param(burst_size,
        "burst_size",
        "Burst size"
        "Number of samples to process in each burst");
    spec.param(num_bursts,
        "num_bursts",
        "Number of bursts"
        "Number of sample bursts to process at once");
    spec.param(num_channels,
        "num_channels",
        "Number of channels",
        "Number of channels to allocate memory for");
}

void HighRatePSD::initialize() {
    holoscan::Operator::initialize();
    make_tensor(outputs,
                {num_channels.get(), num_bursts.get(), burst_size.get()},
                MATX_DEVICE_MEMORY);
    scale_factor = 1.0 / pow(burst_size.get(), 2);
}

void HighRatePSD::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    auto input = op_input.receive<in_t>("in").value();
    auto meta = metadata();
    auto channel_num = meta->get<uint16_t>("channel_number", 0);
    auto out = slice<2>(outputs, {static_cast<index_t>(channel_num), 0, 0},
            {matxDropDim, matxEnd, matxEnd});

    (out = abs2(std::get<0>(input)) * scale_factor).run(std::get<1>(input));
    op_output.emit(
        out_t {
            out,
            std::get<1>(input)
        },
        "out");
}
}  // namespace holoscan::ops
