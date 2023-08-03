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

#include "adv_networking_tx.h"

namespace holoscan::ops {

void AdvConnectorOpTx::setup(OperatorSpec& spec) {
  spec.input<RFChannel>("rf_in");
  spec.output<AdvNetBurstParams>("burst_out");

  spec.param<uint32_t>(batch_size_,
    "batch_size",
    "Batch size",
    "Batch size for each processing epoch", 1000
  );
  spec.param<uint16_t>(payload_size_,
    "payload_size",
    "Payload size",
    "Payload size to send. Does not include <= L4 headers", 1400
  );
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

void AdvConnectorOpTx::initialize() {
  HOLOSCAN_LOG_INFO("AdvConnectorOpTx::initialize()");
  holoscan::Operator::initialize();

  // Compute how many packets sent per array
  samples_per_pkt = (payload_size_.get() - RFPacket::header_size()) / sizeof(complex_t);
  num_packets_buf = packets_per_channel(payload_size_.get(), numPulses.get(), numSamples.get());
  HOLOSCAN_LOG_INFO("samples_per_pkt: {}", samples_per_pkt);
  HOLOSCAN_LOG_INFO("num_packets_buf: {}", num_packets_buf);

  //todo Figure out a better way to break up and send a large chunk of data
  if (num_packets_buf >= batch_size_.get()) {
    HOLOSCAN_LOG_ERROR(
      "RF array size too large: [{}, {}] requires {} packets and the max batch size is set to {}",
      numPulses.get(), numSamples.get(), num_packets_buf, batch_size_.get()
    );
    exit(1);
  }

  // Reserve memory
  buf_stride = RFPacket::packet_size(samples_per_pkt);
  buf_size   = num_packets_buf * buf_stride;
  cudaMallocHost((void **)&mem_buf, buf_size);

  packets_buf = new RFPacket[num_packets_buf];
  for (size_t i = 0; i < num_packets_buf; i++) {
    packets_buf[i] = RFPacket(&mem_buf[i * buf_stride]);
  }

  HOLOSCAN_LOG_INFO("AdvConnectorOpTx::initialize() complete");
}

void AdvConnectorOpTx::compute(InputContext& op_input,
                                   OutputContext& op_output,
                                   ExecutionContext& context) {
  HOLOSCAN_LOG_INFO("AdvConnectorOpTx::compute()");
  AdvNetStatus ret;

  // Input is pulse/sample data from a single channel
  auto rf_data = op_input.receive<RFChannel>("rf_in");

  /**
   * Spin waiting until a buffer is free. This can be stalled by sending
   * faster than the NIC can handle it. We expect the transmit operator to
   * operate much faster than the receiver since it's not having to do any
   * work to construct packets, and just copying from a buffer into memory.
  */
  while (!adv_net_tx_burst_available(num_packets_buf)) {}

  auto msg = CreateSharedBurstParams();
  adv_net_set_hdr(msg, port_id, queue_id, num_packets_buf);

  if ((ret = adv_net_get_tx_pkt_burst(msg)) != AdvNetStatus::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Error returned from adv_net_get_tx_pkt_burst: {}",
      static_cast<int>(ret));
    return;
  }

  // Generate packets from RF data //todo Optimize this process
  index_t ix_buf = 0;
  index_t ix_max = static_cast<index_t>(numSamples.get());
  for (index_t ix_pulse = 0; ix_pulse < numPulses.get(); ix_pulse++) {
    for (index_t ix_sample = 0; ix_sample < numSamples.get(); ix_sample += samples_per_pkt) {
      // Slice to the samples this packet will send
      auto data = rf_data->data.Slice<1>(
        {ix_pulse, ix_sample},
        {matxDropDim, std::min(ix_sample + samples_per_pkt, ix_max)}
      );

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
  const bool send_eoa_msg = rf_data->channel_id == (numChannels.get() - 1);
  if (send_eoa_msg) {
    packets_buf[num_packets_buf - 1].set_end_array(1);
  }

  // Transmit
  for (int pkt_idx = 0; pkt_idx < msg->hdr.num_pkts; pkt_idx++) {
    if ((ret = adv_net_set_cpu_udp_payload(
      msg,
      pkt_idx,
      packets_buf[pkt_idx].get_ptr(),
      buf_stride
    )) != AdvNetStatus::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to create packet {}", pkt_idx);
    }
  }
  HOLOSCAN_LOG_INFO("AdvConnectorOpTx sending {} packets...", msg->hdr.num_pkts);
  op_output.emit(msg, "burst_out");

  HOLOSCAN_LOG_INFO("AdvConnectorOpTx::compute() done");
}

}  // namespace holoscan::ops