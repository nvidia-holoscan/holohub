/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "source.h"

#include <chrono>

namespace holoscan::ops {

// ----- TargetSimulator ------------------------------------------------------
void TargetSimulator::setup(OperatorSpec& spec) {
  spec.output<std::shared_ptr<RFChannel>>("rf_out");
  spec.param(data_rate, "data_rate",
              "Data rate to generate data (Gbps)",
              "Operates by sleeping before emitting a single channel", {});
  spec.param(num_transmits, "num_transmits",
              "Number of waveform transmissions",
              "Number of waveform transmissions to simulate", {});
  spec.param(num_pulses, "num_pulses",
              "Number of pulses",
              "Number of pulses per channel", {});
  spec.param(num_channels,
              "num_channels",
              "Number of channels",
              "Number of channels", {});
  spec.param(waveform_length,
              "waveform_length",
              "Waveform length",
              "Length of waveform", {});
  spec.param(num_samples,
              "num_samples",
              "Number of samples",
              "Number of samples per channel", {});
}

void TargetSimulator::initialize() {
  HOLOSCAN_LOG_INFO("TargetSimulator::initialize()");
  holoscan::Operator::initialize();

  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  transmit_count = 0;
  channel_idx = 0;

  /**
   * This is a rudimentary formula to account for the fact that there is unaccounted for
   * processing outside of this Operator. Without it, the true data rate when set to higher
   * values will skew low.
   */
  double modifier = 1;
  int lowest_realtime = 20;  // Lowest data rate modifier is necessary
  if (data_rate.get() > lowest_realtime) {
    double diff = 0.0015 * static_cast<double>(data_rate.get() - lowest_realtime);
    modifier = modifier > diff ? modifier - diff : 0;
  }

  // Compute how long to sleep to achieve desired data rate
  double bits_per_channel = 8 * num_samples.get() * num_pulses.get() * sizeof(complex_t);
  double tsleep_sec = bits_per_channel / static_cast<double>(1e9 * data_rate.get());
  tsleep_us = static_cast<int>(modifier * tsleep_sec * 1e6);

  // Initialize tensor
  make_tensor(simSignal, {num_channels.get(), num_pulses.get(), num_samples.get()});
  simSignal.PrefetchDevice(stream);
  HOLOSCAN_LOG_INFO("TargetSimulator::initialize() done");
}

void TargetSimulator::compute(InputContext&,
                              OutputContext& op_output,
                              ExecutionContext& context) {
  if (transmit_count == num_transmits.get()) {
    op_output.emit(nullptr, "rf_out");
    return;
  }

  std::chrono::steady_clock::time_point sim_start = std::chrono::steady_clock::now();

  HOLOSCAN_LOG_INFO("TargetSimulator::compute() - simulation {} of {}, channel {} of {}",
    transmit_count+1, num_transmits.get(), channel_idx+1, num_channels.get());

  if (channel_idx == 0) {
    // todo Simulate targets
    auto sig_real = simSignal.RealView();
    auto sig_imag = simSignal.ImagView();
    (sig_real = sig_real + ones(sig_real.Shape())).run(stream);
    (sig_imag = sig_imag + 2 * ones(sig_imag.Shape())).run(stream);
  }

  // Subtract time spent simulating from channel sleep time
  std::chrono::steady_clock::time_point sim_end = std::chrono::steady_clock::now();
  auto sim_dt = std::chrono::duration_cast<std::chrono::microseconds>(sim_end - sim_start).count();
  if (tsleep_us > sim_dt) {
    usleep(tsleep_us - sim_dt);
  }

  auto channel_data = simSignal.Slice<2>({channel_idx, 0, 0},
                                         {matxDropDim, matxEnd, matxEnd});
  auto params = std::make_shared<RFChannel>(channel_data, transmit_count, channel_idx, stream);
  op_output.emit(params, "rf_out");

  channel_idx++;
  if (channel_idx == num_channels.get()) {
    // Sent all channels, move to next array
    transmit_count++;
    channel_idx = 0;
  }

  HOLOSCAN_LOG_INFO("TargetSimulator::compute() done");
}

}  // namespace holoscan::ops
