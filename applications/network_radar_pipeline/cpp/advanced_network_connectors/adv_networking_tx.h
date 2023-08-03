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
#include "adv_network_tx.h"

#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <arpa/inet.h>

namespace holoscan::ops {

class AdvConnectorOpTx : public Operator {
public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvConnectorOpTx)

  AdvConnectorOpTx() = default;
  ~AdvConnectorOpTx() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;

private:
  // Radar settings
  Parameter<uint16_t> numPulses;
  Parameter<uint16_t> numSamples;
  Parameter<uint16_t> waveformLength;
  Parameter<uint16_t> numChannels;

  static constexpr uint16_t port_id = 0;
  static constexpr uint16_t queue_id = 0;
  Parameter<uint32_t> batch_size_;
  Parameter<uint16_t> payload_size_;

  index_t samples_per_pkt;
  index_t num_packets_buf;
  size_t buf_stride;
  size_t buf_size;
  RFPacket *packets_buf;
  uint8_t *mem_buf;

}; // AdvConnectorOpTx

}  // namespace holoscan::ops