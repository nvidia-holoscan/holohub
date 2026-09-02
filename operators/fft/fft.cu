// SPDX-FileCopyrightText: 2024-2026 Voyager Technologies, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "fft.hpp"

#include <stdexcept>
#include <utility>

using in_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;
using out_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

namespace holoscan::ops {

void FFT::setup(OperatorSpec& spec) {
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
    spec.param(spectrum_type,
        "spectrum_type",
        "Spectrum type",
        "VITA 49.2 spectrum type to pass along in metadata");
    spec.param(averaging_type,
        "averaging_type",
        "Averaging type",
        "VITA 49.2 averaging type to pass along in metadata");
    spec.param(window_time,
        "window_time",
        "Window time",
        "VITA 49.2 window time to pass along in metadata");
    spec.param(window_type,
        "window_type",
        "Window type",
        "VITA 49.2 window type to pass along in metadata");
    spec.param(transform_points,
        "transform_points",
        "Transform points",
        "Number of FFT points to take and VITA 49.2 transform points to pass along in metadata");
    spec.param(window_points,
        "window_points",
        "Window points",
        "VITA 49.2 window points to pass along in metadata");
    spec.param(resolution,
        "resolution",
        "Resolution",
        "VITA 49.2 resolution to pass along in metadata");
    spec.param(span,
        "span",
        "Span",
        "VITA 49.2 span to pass along in metadata");
    spec.param(weighting_factor,
        "weighting_factor",
        "Weighting factory",
        "VITA 49.2 weighting factor to pass along in metadata");
    spec.param(f1_index,
        "f1_index",
        "F1 index",
        "VITA 49.2 F1 index to pass along in metadata");
    spec.param(f2_index,
        "f2_index",
        "F2 index",
        "VITA 49.2 F2 index to pass along in metadata");
    spec.param(window_time_delta,
        "window_time_delta",
        "Window time delta",
        "VITA 49.2 window time delta to pass along in metadata");
}

void FFT::initialize() {
    holoscan::Operator::initialize();
    if (burst_size.get() <= 0 || num_bursts.get() <= 0 || num_channels.get() == 0) {
        throw std::runtime_error("fft.burst_size, num_bursts, and num_channels must all be > 0");
    }
}

void FFT::compute(InputContext& op_input, OutputContext& op_output, ExecutionContext& context) {
    auto input = op_input.receive<in_t>("in").value();
    auto input_tensor = std::get<0>(input);
    auto stream = std::get<1>(input);
    auto meta = metadata();
    auto channel_num = meta->get<uint16_t>("channel_number", 0);
    if (channel_num >= num_channels.get()) {
        throw std::runtime_error(fmt::format(
            "Invalid channel_number {} for fft.num_channels {}", channel_num, num_channels.get()));
    }
    if (input_tensor.Size(0) != num_bursts.get()
        || input_tensor.Size(1) != burst_size.get()) {
        throw std::runtime_error(fmt::format(
            "FFT input shape ({}, {}) must match configured shape ({}, {})",
            input_tensor.Size(0),
            input_tensor.Size(1),
            num_bursts.get(),
            burst_size.get()));
    }

    // Every emitted message owns its output storage. Reusing one tensor per
    // channel can overwrite data while another branch still consumes it.
    auto out = make_tensor<complex>(
        {static_cast<index_t>(num_bursts.get()), static_cast<index_t>(burst_size.get())},
        MATX_ASYNC_DEVICE_MEMORY,
        stream);

    (out = fftshift1D(fft(input_tensor))).run(stream);

    if (spectrum_type.has_value())
        meta->set("spectrum_type", spectrum_type.get());
    if (averaging_type.has_value())
        meta->set("averaging_type", averaging_type.get());
    if (window_time.has_value())
        meta->set("window_time_delta_interpretation", window_time.get());
    if (window_type.has_value())
        meta->set("window_type", window_type.get());
    if (transform_points.has_value())
        meta->set("num_transform_points", transform_points.get());
    if (window_points.has_value())
        meta->set("num_window_points", window_points.get());
    if (resolution.has_value())
        meta->set("resolution", resolution.get());
    if (span.has_value())
        meta->set("span", span.get());
    if (weighting_factor.has_value())
        meta->set("weighting_factor", weighting_factor.get());
    if (f1_index.has_value())
        meta->set("f1_index", f1_index.get());
    if (f2_index.has_value())
        meta->set("f2_index", f2_index.get());
    if (window_time_delta.has_value())
        meta->set("window_time_delta", window_time_delta.get());

    op_output.emit(
        out_t {
            std::move(out),
            stream
        },
        "out");
}
}  // namespace holoscan::ops
