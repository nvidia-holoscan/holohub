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

namespace holoscan::ops {

// ----- TargetSimulator ------------------------------------------------------
void TargetSimulator::setup(OperatorSpec& spec) {
  spec.output<std::shared_ptr<RFChannel>>("rf_out");
  spec.param(numTransmits, "numTransmits",
              "Number of waveform transmissions",
              "Number of waveform transmissions to simulate", {});
  spec.param(numPulses, "numPulses",
              "Number of pulses",
              "Number of pulses per channel", {});
  spec.param(numChannels,
              "numChannels",
              "Number of channels",
              "Number of channels", {});
  spec.param(waveformLength,
              "waveformLength",
              "Waveform length",
              "Length of waveform", {});
  spec.param(numSamples,
              "numSamples",
              "Number of samples",
              "Number of samples per channel", {});
}

void TargetSimulator::initialize() {
  HOLOSCAN_LOG_INFO("TargetSimulator::initialize()");
  holoscan::Operator::initialize();

  cudaStreamCreate(&stream);
  transmit_count = 0;
  channel_idx = 0;

  // Initialize tensor
  simSignal = new tensor_t<complex_t, 3>(
    {numChannels.get(), numPulses.get(), numSamples.get()});

  simSignal->PrefetchDevice(stream);
  HOLOSCAN_LOG_INFO("TargetSimulator::initialize() done");
}

void TargetSimulator::compute(InputContext&,
                              OutputContext& op_output,
                              ExecutionContext& context) {
  HOLOSCAN_LOG_INFO("TargetSimulator::compute() - simulation {} of {}, channel {} of {}",
    transmit_count+1, numTransmits.get(), channel_idx+1, numChannels.get());

  if (channel_idx == 0) {
    // todo Simulate targets
    auto sig_real = simSignal->RealView();
    auto sig_imag = simSignal->ImagView();
    (sig_real = sig_real + ones(sig_real.Shape())).run(stream);
    (sig_imag = sig_imag + 2 * ones(sig_imag.Shape())).run(stream);
  }

  auto channel_data = simSignal->Slice<2>({channel_idx, 0, 0},
                                          {matxDropDim, matxEnd, matxEnd});
  auto params = std::make_shared<RFChannel>(channel_data, transmit_count, channel_idx, stream);
  op_output.emit(params, "rf_out");

  channel_idx++;
  if (channel_idx == numChannels.get()) {
    // Sent all channels, move to next array
    transmit_count++;
    channel_idx = 0;
    if (transmit_count == numTransmits.get()) {
      GxfGraphInterrupt(context.context());
    }
  }

  HOLOSCAN_LOG_INFO("TargetSimulator::compute() done");
}

}  // namespace holoscan::ops
