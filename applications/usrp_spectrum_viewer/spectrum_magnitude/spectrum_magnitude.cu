// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "spectrum_magnitude.hpp"

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
    make_tensor(outputs_,  {num_channels, burst_size}, MATX_DEVICE_MEMORY);
    make_tensor(abs2_buf_,
                {num_channels, num_bursts, burst_size},
                MATX_DEVICE_MEMORY);
    scale_factor_ = 1.0f / static_cast<float>(burst_size * burst_size);
}

void SpectrumMagnitudeOp::compute(InputContext& op_input,
                                  OutputContext& op_output,
                                  ExecutionContext&) {
    auto input = op_input.receive<in_t>("in").value();
    auto meta = metadata();
    auto channel_num = meta->get<uint16_t>("channel_number", 0);
    auto out = slice<1>(outputs_, {static_cast<index_t>(channel_num), 0},
            {matxDropDim, matxEnd});

    // Keep the per-burst power values so we can average a full FFT batch down
    // to one displayable PSD curve per channel.
    auto abs2_slice = slice<2>(abs2_buf_,
        {static_cast<index_t>(channel_num), 0, 0},
        {matxDropDim, matxEnd, matxEnd});
    (abs2_slice = abs2(std::get<0>(input)) * scale_factor_).run(std::get<1>(input));

    // Convert average linear power to dB for visualization.
    // These are lazy MatX expressions, so they fuse into one kernel at .run().
    const float inv_num_averages = 1.0f / static_cast<float>(num_averages_.get());
    const auto averaged_power = sum(abs2_slice, {0}) * inv_num_averages;
    const auto power_db = 10.0f * log10(averaged_power);
    (out = power_db).run(std::get<1>(input));

    meta->set("num_averages", num_averages_.get());

    op_output.emit(
        out_t {
            out,
            std::get<1>(input)
        },
        "out");
}
}  // namespace holoscan::ops
