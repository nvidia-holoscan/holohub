// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "low_rate_psd.hpp"

using in_t = std::tuple<tensor_t<float, 2>, cudaStream_t>;
using out_t = tensor_t<int8_t, 1>;

namespace holoscan::ops {

void LowRatePSD::setup(OperatorSpec& spec) {
    spec.input<in_t>("in");
    spec.output<out_t>("out");
    spec.param(burst_size,
        "burst_size",
        "Burst size",
        "Number of samples to process at once");
    spec.param(num_averages,
        "num_averages",
        "Number of averages",
        "Number of averages to take and pass along in metadata");
    spec.param(num_channels,
        "num_channels",
        "Number of channels",
        "Number of channels to allocate memory for");
}

void LowRatePSD::initialize() {
    holoscan::Operator::initialize();
    make_tensor(outputs, {num_channels.get(), burst_size.get()}, MATX_DEVICE_MEMORY);
    make_tensor(maxima, {burst_size.get()}, MATX_DEVICE_MEMORY);
    make_tensor(minima, {burst_size.get()}, MATX_DEVICE_MEMORY);
    (maxima = 127.0).run();
    (minima = -128.0).run();
}

void LowRatePSD::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    auto input = op_input.receive<in_t>("in").value();
    auto meta = metadata();
    auto channel_num = meta->get<uint16_t>("channel_number", 0);
    auto out = slice<1>(outputs, {static_cast<index_t>(channel_num), 0},
            {matxDropDim, matxEnd});

    (out = as_int8(
        min(max(
            10.0 * log10(
                sum(std::get<0>(input), {0}) * (1.0 / (float)num_averages.get())),
            minima), maxima))).run(std::get<1>(input));

    meta->set("num_averages", num_averages.get());

    op_output.emit(out_t {out}, "out");
}
}  // namespace holoscan::ops
