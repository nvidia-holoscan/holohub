// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "low_rate_psd.hpp"

namespace holoscan::ops {

void LowRatePSD::setup(OperatorSpec& spec) {
    spec.input<tensor_t<float, 1>>("in");
    spec.output<tensor_t<int8_t, 1>>("out");
    spec.param(burst_size,
               "burst_size",
               "Burst size",
               "Number of samples to process at once");
    spec.param(num_averages,
               "num_averages",
               "Number of averages",
               "Number of averages to take and pass along in metadata");
}

void LowRatePSD::initialize() {
    holoscan::Operator::initialize();
    make_tensor(psdAccumulator, {burst_size.get()}, MATX_DEVICE_MEMORY);
    make_tensor(psdAverage, {burst_size.get()}, MATX_DEVICE_MEMORY);
    reset();
}

void LowRatePSD::reset() {
    fft_count = 0;
    (psdAccumulator = zeros(psdAccumulator.Shape())).run();
}

void LowRatePSD::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) {
    auto high_rate_psd = op_input.receive<tensor_t<float, 1>>("in").value();
    auto new_meta = metadata();

    if (new_meta->get("context_field_change", false)) {
        reset();
    }

    if (fft_count == 0) {
        current_meta.clear();
        current_meta.update(*new_meta);
        current_meta.set("num_averages", num_averages.get());
    }

    // Accumulate the high-rate PSD data
    (psdAccumulator += high_rate_psd).run();
    fft_count++;

    // Compute the average and emit
    if (fft_count == num_averages.get()) {
        auto meta = metadata();
        meta->clear();
        meta->update(current_meta);
        (psdAverage = as_int8(10 * log10(psdAccumulator * (1 / (float)num_averages.get())))).run();
        op_output.emit(psdAverage, "out");
        reset();
    }
}
}  // namespace holoscan::ops
