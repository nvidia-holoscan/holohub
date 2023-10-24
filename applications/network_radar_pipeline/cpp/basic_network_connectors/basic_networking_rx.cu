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
#include "basic_networking_rx.h"

namespace holoscan::ops {

void BasicConnectorOpRx::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<NetworkOpBurstParams>>("burst_in");
  spec.output<std::shared_ptr<RFArray>>("rf_out");
  spec.param(max_pkts, "batch_size",
              "Max packets",
              "Max number of packets received", {});
  spec.param(payload_size, "max_payload_size",
              "Max payload size in bytes",
              "Max payload size in bytes for received packets", {});
  spec.param(numTransmits, "numTransmits",
              "Number of waveform transmissions",
              "Number of waveform transmissions to simulate", {});
  spec.param(bufferSize, "bufferSize",
              "Size of RF buffer",
              "Max number of transmits that can be held at once", {});
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

void BasicConnectorOpRx::initialize() {
  holoscan::Operator::initialize();
  num_rx = 0;
  buffer_track = BasicBufferTracking(bufferSize.get());
  pkt_buf = new RFPacket[max_pkts];
  rf_data = new tensor_t<complex_t, 4>(
    {bufferSize.get(), numChannels.get(), numPulses.get(), numSamples.get()});

  // Compute number of packets expected per array
  samples_per_arr = numPulses.get() * numChannels.get() * numSamples.get();
  pkts_per_arr = packets_per_array(payload_size.get(),
                                    numPulses.get(),
                                    numChannels.get(),
                                    numSamples.get());

  cudaStreamCreate(&stream);

  HOLOSCAN_LOG_INFO("Expecting to receive {} packets", numTransmits.get() * pkts_per_arr);
}

void BasicConnectorOpRx::compute(InputContext& op_input,
                               OutputContext& op_output,
                               ExecutionContext& context) {
  auto in = op_input.receive<std::shared_ptr<NetworkOpBurstParams>>("burst_in").value();
  num_rx += in->num_pkts;

  uint8_t *buf_ptr = in->data;
  for (size_t i = 0; i < in->num_pkts; i++) {
    // Get packet and adjust pointer
    pkt_buf[i] = RFPacket(buf_ptr);
    buf_ptr += payload_size;  // RFPacket::packet_size(pkt_buf[i].get_num_samples());

    // Make sure this isn't wrapping the buffer - drop if it is
    if ((pkt_buf[i].get_waveform_id() >= buffer_track.pos + bufferSize.get()) ||
        (pkt_buf[i].get_waveform_id() < buffer_track.pos)) {
      HOLOSCAN_LOG_ERROR("Waveform ID {} exceeds buffer limits (pos: {}, size: {}), dropping",
        pkt_buf[i].get_waveform_id(), buffer_track.pos, bufferSize.get());
    } else {  // Copy into rf_data
      const index_t buffer_idx  = pkt_buf[i].get_waveform_id() % bufferSize.get();
      const index_t channel_idx = pkt_buf[i].get_channel_idx();
      const index_t pulse_idx   = pkt_buf[i].get_pulse_idx();
      const index_t sample_idx  = pkt_buf[i].get_sample_idx();
      const uint16_t end_array  = pkt_buf[i].get_end_array();

      // Mark if we've received the end-of-array message
      if (end_array) {
        buffer_track.received_end[buffer_idx] = true;
      }

      buffer_track.sample_cnt[buffer_idx] += pkt_buf[i].get_num_samples();
      auto rf_tensor = rf_data->Slice<1>(
        {buffer_idx, channel_idx, pulse_idx, sample_idx},
        {matxDropDim, matxDropDim, matxDropDim, sample_idx + pkt_buf[i].get_num_samples()});
      const size_t n_bytes = RFPacket::payload_size(pkt_buf[i].get_num_samples());
      cudaMemcpy(rf_tensor.Data(), pkt_buf[i].data(), n_bytes, cudaMemcpyHostToDevice);
    }

    if ((num_rx - in->num_pkts + i) % 1000 == 0) {
      HOLOSCAN_LOG_INFO("Packet: [{}, {}, {}, {} - {}] ({} total, {} / {})",
        pkt_buf[i].get_waveform_id(),
        pkt_buf[i].get_pulse_idx(),
        pkt_buf[i].get_channel_idx(),
        pkt_buf[i].get_sample_idx(),
        pkt_buf[i].get_num_samples(),
        num_rx,
        buffer_track.sample_cnt[buffer_track.pos_wrap],
        samples_per_arr);
    }
  }

  delete[] in->data;

  // Check if we can emit an array
  if (buffer_track.is_ready(samples_per_arr)) {
    auto params = std::make_shared<RFArray>(
      rf_data->Slice<3>(
        {static_cast<index_t>(buffer_track.pos_wrap), 0, 0, 0},
        {matxDropDim, matxEnd, matxEnd, matxEnd}),
      0, stream);
    op_output.emit(params, "rf_out");
    HOLOSCAN_LOG_INFO("Emitting {} with {}/{} samples",
      buffer_track.pos,
      buffer_track.sample_cnt[buffer_track.pos_wrap],
      samples_per_arr);
    buffer_track.increment();
  }
}

}  // namespace holoscan::ops
