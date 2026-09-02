// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "spectrum_magnitude.hpp"

#include <stdexcept>
#include <utility>

using namespace matx;

using in_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;
using out_t = std::tuple<tensor_t<float, 1>, cudaStream_t>;

namespace holoscan::ops {

void SpectrumMagnitudeOp::setup(OperatorSpec& spec) {
    spec.input<in_t>("in");
    spec.output<out_t>("out");
    spec.param(burst_size_,
        "burst_size",
        "Burst size",
        "Number of samples to process at once");
    spec.param(num_bursts_,
        "num_bursts",
        "Number of bursts",
        "Number of sample bursts to process at once");
    spec.param(num_averages_,
        "num_averages",
        "Number of averages",
        "Number of averages to take and pass along in metadata");
    spec.param(num_channels_,
        "num_channels",
        "Number of channels",
        "Number of channels to allocate memory for");
}

void SpectrumMagnitudeOp::initialize() {
    holoscan::Operator::initialize();

     const int burst_size = burst_size_.get();
     const int num_bursts = num_bursts_.get();
     const auto num_channels = num_channels_.get();
     const auto num_averages = num_averages_.get();
     if (burst_size <= 0 || num_bursts <= 0 || num_channels == 0 || num_averages == 0) {
         throw std::runtime_error(
             "spectrum_magnitude.burst_size, num_bursts, num_channels, and num_averages must all "
             "be > 0");
     }
     if (num_averages != static_cast<uint32_t>(num_bursts)) {
         throw std::runtime_error(
             "spectrum_magnitude.num_averages must equal num_bursts; the batch is averaged over "
             "all num_bursts rows");
     }
    scale_factor_ = 1.0f / (static_cast<float>(burst_size) * static_cast<float>(burst_size));
}

void SpectrumMagnitudeOp::compute(InputContext& op_input,
                                  OutputContext& op_output,
                                  ExecutionContext&) {
    auto input = op_input.receive<in_t>("in").value();
    auto input_tensor = std::get<0>(input);
    auto stream = std::get<1>(input);
    auto meta = metadata();
    auto channel_num = meta->get<uint16_t>("channel_number", 0);
    if (channel_num >= num_channels_.get()) {
        throw std::runtime_error(fmt::format(
            "Invalid channel_number {} for spectrum_magnitude.num_channels {}",
            channel_num,
            num_channels_.get()));
    }
    if (input_tensor.Size(0) != num_bursts_.get()
        || input_tensor.Size(1) != burst_size_.get()) {
        throw std::runtime_error(fmt::format(
            "Spectrum magnitude input shape ({}, {}) must match configured shape ({}, {})",
            input_tensor.Size(0),
            input_tensor.Size(1),
            num_bursts_.get(),
            burst_size_.get()));
    }

    auto out = make_tensor<float>({static_cast<index_t>(burst_size_.get())},
                                  MATX_ASYNC_DEVICE_MEMORY,
                                  stream);

    // Keep the per-burst power values so we can average a full FFT batch down
    // to one displayable PSD curve per channel.
    auto abs2_buffer = make_tensor<float>(
        {static_cast<index_t>(num_bursts_.get()), static_cast<index_t>(burst_size_.get())},
        MATX_ASYNC_DEVICE_MEMORY,
        stream);
    (abs2_buffer = abs2(input_tensor) * scale_factor_).run(stream);

    // Convert average linear power to dB for visualization.
    // These are lazy MatX expressions, so they fuse into one kernel at .run().
    const float inv_num_averages = 1.0f / static_cast<float>(num_averages_.get());
    const auto averaged_power = sum(abs2_buffer, {0}) * inv_num_averages;
    const auto power_db = 10.0f * log10(averaged_power);
    (out = power_db).run(stream);

    meta->set("num_averages", num_averages_.get());

    op_output.emit(
        out_t {
            std::move(out),
            stream
        },
        "out");
}
}  // namespace holoscan::ops
