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
#include "basic_network_operator_rx.h"

// Tracks the status of filling an RF array
struct BasicBufferTracking {
  size_t pos;
  size_t pos_wrap;
  size_t buffer_size;
  std::vector<bool> received_end;
  std::vector<size_t> sample_cnt;

  BasicBufferTracking() = default;
  explicit BasicBufferTracking(const size_t _buffer_size)
    : pos(0),
      pos_wrap(0),
      buffer_size(_buffer_size),
      received_end(_buffer_size, false),
      sample_cnt(_buffer_size, 0) {}

  void increment() {
    received_end[pos_wrap] = false;
    sample_cnt[pos_wrap] = 0;
    pos++;
    pos_wrap = pos % buffer_size;
  }

  bool is_ready(const size_t samples_per_arr) {
    return received_end[pos_wrap] || sample_cnt[pos_wrap] >= samples_per_arr;
  }
};

namespace holoscan::ops {

class BasicConnectorOpRx : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicConnectorOpRx)

  BasicConnectorOpRx() = default;
  ~BasicConnectorOpRx() {
    if (pkt_buf) {
      delete pkt_buf;
    }
    if (rf_data) {
      delete rf_data;
    }
  }

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;

 private:
  int num_rx;
  Parameter<uint16_t> max_pkts;
  Parameter<uint16_t> payload_size;
  Parameter<uint16_t> num_transmits;
  Parameter<uint16_t> buffer_size;
  Parameter<uint16_t> num_pulses;
  Parameter<uint16_t> num_samples;
  Parameter<uint16_t> waveform_length;
  Parameter<uint16_t> num_channels;
  RFPacket *pkt_buf;
  size_t samples_per_arr;
  size_t pkts_per_arr;
  BasicBufferTracking buffer_track;
  tensor_t<complex_t, 4> *rf_data = nullptr;
  cudaStream_t stream;
};  // BasicConnectorOpRx

}  // namespace holoscan::ops
