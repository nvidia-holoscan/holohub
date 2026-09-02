// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "log.hpp"

#include <stdexcept>

using namespace matx;

namespace holoscan::ops {

void LogOp::setup(OperatorSpec& spec) {
  spec.input<in_t>("in");
  spec.param(num_channels_, "num_channels",
      "Number of Channels",
      "The number of RF channels being processed.", 1);
  spec.param(log_interval_, "log_interval",
      "Log Interval",
      "Interval in seconds to log the data rate statistics.", 5);
  spec.param(log_data_, "log_data",
      "Log Data",
      "If true, log detailed data information for debugging.", false);
}

void LogOp::initialize() {
  Operator::initialize();
  if (num_channels_.get() <= 0 || log_interval_.get() <= 0) {
    throw std::runtime_error("log.num_channels and log_interval must both be > 0");
  }
  const auto num_channels = static_cast<size_t>(num_channels_.get());
  total_samples_.resize(num_channels, 0);
  start_.resize(num_channels);
  elapsed_.resize(num_channels, std::chrono::steady_clock::duration::zero());
}

void LogOp::compute(InputContext& op_input,
    OutputContext&,
    ExecutionContext&) {

  // Receive input tensor and CUDA stream
  auto input = op_input.receive<in_t>("in").value();
  auto tensor = std::get<0>(input);

  // Access metadata
  auto meta = metadata();
  auto channel_num = meta->get<uint16_t>("channel_number", 0);
  if (channel_num >= static_cast<uint16_t>(total_samples_.size())) {
    HOLOSCAN_LOG_ERROR(
        "Invalid channel_number {} (expected < {})", channel_num, total_samples_.size());
    return;
  }

  // Get timing information
  auto now = std::chrono::steady_clock::now();
  if (start_[channel_num] == std::chrono::steady_clock::time_point{}) {
    // Establish the first arrival time without counting radio/application
    // startup latency as processing time.
    start_[channel_num] = now;
  } else {
    const auto interval = now - start_[channel_num];
    start_[channel_num] = now;

    // The incoming tensor is FFT output, but the throughput estimate is
    // expressed in terms of the original interleaved sc16 IQ stream.
    const int64_t num_samples =
        static_cast<int64_t>(tensor.Size(0)) * static_cast<int64_t>(tensor.Size(1));
    total_samples_[channel_num] += num_samples;
    elapsed_[channel_num] += interval;
  }

  // Log statistics
  auto seconds = std::chrono::duration<double>(elapsed_[channel_num]).count();
  if (total_samples_[channel_num] > 0 && seconds >= log_interval_.get()) {
    auto total_bits = total_samples_[channel_num] * sizeof(int16_t) * 2 * 8;
    HOLOSCAN_LOG_INFO("Processed {} samples from channel {} at {:.2f} MSps ({:.2f} Gbps)",
        total_samples_[channel_num],
        channel_num,
        total_samples_[channel_num] / seconds / 1e6,
        total_bits / seconds / 1e9);
    total_samples_[channel_num] = 0;
    elapsed_[channel_num] = std::chrono::steady_clock::duration::zero();
  }

  // Log data for debugging
  if (log_data_.get()) {
    HOLOSCAN_LOG_INFO("Received tensor from channel {} with rank {} and shape: ({}, {})",
        channel_num, tensor.Rank(), tensor.Size(0), tensor.Size(1));
    HOLOSCAN_LOG_INFO("Every 20th sample of the first burst of channel {}:", channel_num);
    set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
    print(slice<1>(tensor, {0, 0}, {matxDropDim, matxEnd}, {1, 20}));
  }
}

}  // namespace holoscan::ops
