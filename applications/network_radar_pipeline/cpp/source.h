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
#pragma once

#include "common.h"

namespace holoscan::ops {

class TargetSimulator : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(TargetSimulator)

  TargetSimulator() = default;
  ~TargetSimulator() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override;

 private:
  cudaStream_t stream;
  Parameter<int> data_rate;
  Parameter<uint16_t> num_transmits;
  Parameter<uint16_t> num_pulses;
  Parameter<uint16_t> num_samples;
  Parameter<uint16_t> waveform_length;
  Parameter<uint16_t> num_channels;
  Parameter<uint16_t> samplesPerPkt;
  index_t transmit_count;
  index_t channel_idx;
  int tsleep_us;

  tensor_t<complex_t, 3> simSignal;
};

};  // namespace holoscan::ops
