// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "log.hpp"

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
  total_samples_.resize(num_channels_, 0);
  start_.resize(num_channels_, std::chrono::steady_clock::now());
  elapsed_.resize(num_channels_, std::chrono::steady_clock::duration::zero());
}

void LogOp::compute(InputContext& op_input,
    OutputContext& op_output,
    ExecutionContext& context) {

  // Receive input tensor and CUDA stream
  auto input = op_input.receive<in_t>("in").value();
  auto tensor = std::get<0>(input);
  auto stream = std::get<1>(input);

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
  auto interval = now - start_[channel_num];
  start_[channel_num] = now;

  // The incoming tensor is FFT output, but the throughput estimate is expressed
  // in terms of the original interleaved sc16 IQ stream for easier SDR tuning.
  auto num_samples = tensor.Size(0) * tensor.Size(1);

  // Update statistics
  total_samples_[channel_num] += num_samples;
  elapsed_[channel_num] += interval;

  // Log statistics
  auto seconds = std::chrono::duration<double>(elapsed_[channel_num]).count();
  if (total_samples_[channel_num] > 0 && seconds >= log_interval_) {
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
  if (log_data_) {
    HOLOSCAN_LOG_INFO("Received tensor from channel {} with rank {} and shape: ({}, {})",
        channel_num, tensor.Rank(), tensor.Size(0), tensor.Size(1));
    HOLOSCAN_LOG_INFO("Every 20th sample of the first burst of channel {}:", channel_num);
    set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
    print(slice<1>(tensor, {0, 0}, {matxDropDim, matxEnd}, {1, 20}));
  }
}

}  // namespace holoscan::ops
