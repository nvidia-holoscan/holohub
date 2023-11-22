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
#include "basic_networking_tx.h"

namespace holoscan::ops {

void BasicConnectorOpTx::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<RFChannel>>("rf_in");
  spec.output<std::shared_ptr<NetworkOpBurstParams>>("burst_out");
  spec.param(payload_size, "max_payload_size",
              "Max payload size in bytes",
              "Max payload size in bytes for received packets", {});
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

void BasicConnectorOpTx::initialize() {
  HOLOSCAN_LOG_INFO("BasicConnectorOpTx::initialize()");
  holoscan::Operator::initialize();

  // Compute how many packets sent per array
  samples_per_pkt = (payload_size.get() - RFPacket::header_size()) / sizeof(complex_t);
  num_packets_buf = packets_per_channel(payload_size.get(), num_pulses.get(), num_samples.get());
  HOLOSCAN_LOG_INFO("samples_per_pkt: {}", samples_per_pkt);
  HOLOSCAN_LOG_INFO("num_packets_buf: {}", num_packets_buf);
  packets_buf = new RFPacket[num_packets_buf];
  HOLOSCAN_LOG_INFO("BasicConnectorOpTx::initialize() done");
}

void BasicConnectorOpTx::compute(InputContext& op_input,
                                 OutputContext& op_output,
                                 ExecutionContext&) {
  // Input is pulse/sample data from a single channel
  auto rf_data = op_input.receive<std::shared_ptr<RFChannel>>("rf_in").value();
  if (rf_data == nullptr) {
    return;
  }
  HOLOSCAN_LOG_INFO("BasicConnectorOpTx::compute()");

  // Determine buffer size; dependent on whether this is the last channel
  const size_t buf_stride = RFPacket::packet_size(samples_per_pkt);
  const size_t buf_size   = num_packets_buf * buf_stride;

  // Reserve memory
  mem_buf = new uint8_t[buf_size];
  for (size_t i = 0; i < num_packets_buf; i++) {
    packets_buf[i] = RFPacket(&mem_buf[i * buf_stride]);
  }

  // Generate packets from RF data //todo Optimize this process
  index_t ix_buf = 0;
  index_t ix_max = static_cast<index_t>(num_samples.get());
  for (index_t ix_pulse = 0; ix_pulse < num_pulses.get(); ix_pulse++) {
    for (index_t ix_sample = 0; ix_sample < num_samples.get(); ix_sample += samples_per_pkt) {
      // Slice to the samples this packet will send
      auto data = rf_data->data.Slice<1>(
        {ix_pulse, ix_sample},
        {matxDropDim, std::min(ix_sample + samples_per_pkt, ix_max)});

      // Use accessor functions to set payload
      packets_buf[ix_buf].set_waveform_id(rf_data->waveform_id);
      packets_buf[ix_buf].set_sample_idx(ix_sample);
      packets_buf[ix_buf].set_channel_idx(rf_data->channel_id);
      packets_buf[ix_buf].set_pulse_idx(ix_pulse);
      packets_buf[ix_buf].set_num_samples(data.Size(0));
      packets_buf[ix_buf].set_end_array(0);
      packets_buf[ix_buf].set_payload(data.Data(), rf_data->stream);
      ix_buf++;
    }
  }
  if (num_packets_buf != ix_buf) {
    HOLOSCAN_LOG_ERROR("Not sending expected number of packets");
  }

  // Send end-of-array message if this is the last channel of the transmit
  const bool send_eoa_msg = rf_data->channel_id == num_channels.get() - 1;
  if (send_eoa_msg) {
    packets_buf[num_packets_buf - 1].set_end_array(1);
  }

  // Transmit
  auto value = std::make_shared<NetworkOpBurstParams>(mem_buf, buf_size, 1);
  HOLOSCAN_LOG_INFO("BasicConnectorOpTx sending {} packets...", ix_buf);
  op_output.emit(value, "burst_out");

  HOLOSCAN_LOG_INFO("BasicConnectorOpTx::compute() done");
}

}  // namespace holoscan::ops
