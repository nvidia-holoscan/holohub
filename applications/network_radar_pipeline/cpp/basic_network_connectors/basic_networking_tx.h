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
#include "basic_network_operator_tx.h"

namespace holoscan::ops {

class BasicConnectorOpTx : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicConnectorOpTx)

  BasicConnectorOpTx() = default;
  ~BasicConnectorOpTx() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override;

 private:
  Parameter<uint16_t> payload_size;
  Parameter<uint16_t> num_pulses;
  Parameter<uint16_t> num_samples;
  Parameter<uint16_t> waveform_length;
  Parameter<uint16_t> num_channels;
  index_t samples_per_pkt;
  index_t num_packets_buf;
  RFPacket *packets_buf;
  uint8_t *mem_buf;
};  // BasicConnectorOpTx

}  // namespace holoscan::ops
