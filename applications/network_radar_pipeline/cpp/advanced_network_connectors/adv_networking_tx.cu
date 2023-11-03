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

#include "adv_networking_tx.h"  //todo: Rename networking connectors

namespace holoscan::ops {

void AdvConnectorOpTx::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<RFChannel>>("rf_in");
  spec.output<std::shared_ptr<AdvNetBurstParams>>("burst_out");

  // Advanced network operator parameters
  spec.param<AdvNetConfigYaml>(cfg_,
    "cfg",
    "Configuration",
    "Configuration for the advanced network operator",
    AdvNetConfigYaml());

  // Radar parameters
  spec.param<uint16_t>(num_pulses_,
    "num_pulses",
    "Number of pulses",
    "Number of pulses per channel", {});
  spec.param<uint16_t>(num_channels_,
    "num_channels",
    "Number of channels",
    "Number of channels", {});
  spec.param<uint16_t>(waveform_length_,
    "waveform_length",
    "Waveform length",
    "Length of waveform", {});
  spec.param<uint16_t>(num_samples_,
    "num_samples",
    "Number of samples",
    "Number of samples per channel", {});

  // Networking parameters
  spec.param<uint16_t>(samples_per_packet_,
    "samples_per_packet",
    "Number of complex I/Q samples to send per packet",
    "Payload size computed based on samples/packet, not including <= L4 headers", 128);
  spec.param<uint16_t>(header_size_,
    "header_size",
    "Header size",
    "Header size on each packet from L4 and below", 42);
  spec.param<uint16_t>(udp_src_port_,
    "udp_src_port",
    "UDP source port",
    "UDP source port");
  spec.param<uint16_t>(udp_dst_port_,
    "udp_dst_port",
    "UDP destination port",
    "UDP destination port");
  spec.param<std::string>(ip_src_addr_,
    "ip_src_addr",
    "IP source address",
    "IP source address");
  spec.param<std::string>(ip_dst_addr_,
    "ip_dst_addr",
    "IP destination address",
    "IP destination address");
  spec.param<std::string>(eth_dst_addr_,
    "eth_dst_addr",
    "Ethernet destination address",
    "Ethernet destination address");
  spec.param<uint16_t>(port_id_,
    "port_id",
    "Interface number",
    "Interface number");
}

void AdvConnectorOpTx::initialize() {
  HOLOSCAN_LOG_INFO("AdvConnectorOpTx::initialize()");
  register_converter<holoscan::ops::AdvNetConfigYaml>();
  holoscan::Operator::initialize();

  // Read some parameters from config
  if (cfg_.get().tx_.size() != 1 || cfg_.get().tx_[0].queues_.size() != 1) {
    HOLOSCAN_LOG_ERROR("Currently can only handle 1 Tx queue");
    return;
  }
  payload_size_ = RFPacket::packet_size(samples_per_packet_.get());
  batch_size_   = cfg_.get().tx_[0].queues_[0].common_.batch_size_;
  hds_          = cfg_.get().tx_[0].queues_[0].common_.hds_;
  gpu_direct_   = cfg_.get().tx_[0].queues_[0].common_.gpu_direct_;

  if (gpu_direct_ && hds_ == 0) {
    //todo: GPU-only mode Eth+IP+UDP headers
    HOLOSCAN_LOG_ERROR("Not configured for GPU-only, GPUDirect requires HDS mode for now");
    return;
  }

  // Compute how many packets sent per array
  samples_per_pkt = (payload_size_ - RFPacket::header_size()) / sizeof(complex_t);
  pkt_per_pulse   = packets_per_pulse(payload_size_, num_samples_.get());
  num_packets_buf = packets_per_channel(payload_size_, num_pulses_.get(), num_samples_.get());
  HOLOSCAN_LOG_INFO("samples_per_pkt: {}", samples_per_pkt);
  HOLOSCAN_LOG_INFO("num_packets_buf: {}", num_packets_buf);

  if (num_packets_buf >= batch_size_) {
    //todo: Figure out a better way to break up and send a large chunk of data
    HOLOSCAN_LOG_ERROR(
      "RF array size too large: [{}, {}] requires {} packets and the max batch size is set to {}",
      num_pulses_.get(), num_samples_.get(), num_packets_buf, batch_size_);
    return;
  }

  // Reserve memory
  buf_stride = RFPacket::packet_size(samples_per_pkt);
  buf_size   = num_packets_buf * buf_stride;
  if (!gpu_direct_) {
    // On CPU
    mem_buf_h_ = static_cast<uint8_t *>(malloc(buf_size));
    if (mem_buf_h_ == nullptr) {
      HOLOSCAN_LOG_ERROR("Failed to allocate {} bytes of CPU batch memory", buf_size);
      return;
    }
    packets_buf = new RFPacket[num_packets_buf];
    for (size_t i = 0; i < num_packets_buf; i++) {
      packets_buf[i] = RFPacket(&mem_buf_h_[i * buf_stride]);
    }
  }
  else {
    // On GPU
    for (int n = 0; n < num_concurrent; n++) {
      cudaMallocHost(&gpu_bufs[n], sizeof(uint8_t**) * batch_size_);
      cudaStreamCreate(&streams_[n]);
      cudaEventCreate(&events_[n]);
    }
    HOLOSCAN_LOG_INFO("Initialized {} streams and events", num_concurrent);
  }

  adv_net_format_eth_addr(eth_dst_, eth_dst_addr_.get());
  inet_pton(AF_INET, ip_src_addr_.get().c_str(), &ip_src_);
  inet_pton(AF_INET, ip_dst_addr_.get().c_str(), &ip_dst_);

  // ANO expects host order when setting
  ip_src_ = ntohl(ip_src_);
  ip_dst_ = ntohl(ip_dst_);

  HOLOSCAN_LOG_INFO("AdvConnectorOpTx::initialize() complete");
}

/**
 * @brief Set UDP headers on Host memory
 */
AdvNetStatus AdvConnectorOpTx::set_cpu_hdr(AdvNetBurstParams *msg, const int pkt_idx) {
  AdvNetStatus ret;

  // Set Ethernet header
  if ((ret = adv_net_set_cpu_eth_hdr(msg,
                                     pkt_idx,
                                     eth_dst_)) != AdvNetStatus::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set Ethernet header for packet {}", pkt_idx);
    adv_net_free_all_burst_pkts_and_burst(msg);
    return ret;
  }

  // Remove Eth + IP size
  const auto ip_len = payload_size_ + header_size_.get() - (14 + 20);
  if ((ret = adv_net_set_cpu_ipv4_hdr(msg,
                                      pkt_idx,
                                      ip_len,
                                      17,
                                      ip_src_,
                                      ip_dst_)) != AdvNetStatus::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set IP header for packet {}", 0);
    adv_net_free_all_burst_pkts_and_burst(msg);
    return ret;
  }

  // Set UDP header
  if ((ret = adv_net_set_cpu_udp_hdr(msg,
                                     pkt_idx,
                                     // Remove Eth + IP + UDP headers
                                     payload_size_ + header_size_.get() - (14 + 20 + 8),
                                     udp_src_port_.get(),
                                     udp_dst_port_.get())) != AdvNetStatus::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Failed to set UDP header for packet {}", 0);
    adv_net_free_all_burst_pkts_and_burst(msg);
    return ret;
  }

  return AdvNetStatus::SUCCESS;
}

/**
 * @brief Copy I/Q RF data from tensor to packets
 */
__global__
void populate_packets_kernel(uint8_t **out_ptr,
                             complex_t *rf_data,
                             uint16_t waveform_id,
                             uint16_t channel_idx,
                             uint16_t samples_per_pkt,
                             uint16_t pkt_per_pulse,
                             uint16_t num_channels,
                             uint16_t num_samples,
                             uint16_t offset) {
  int pkt_idx   = blockIdx.x;
  int pulse_idx = blockIdx.y;
  int buf_idx   = pkt_idx + pulse_idx * pkt_per_pulse;
  uint8_t *pkt  = out_ptr[buf_idx] + offset;

  int sample_idx  = samples_per_pkt * pkt_idx;
  int pkt_samples = min(samples_per_pkt, num_samples - sample_idx);

  // Send EOA if this is the last packet of the array (i.e. the last packet of the
  // last pulse of the last channel)
  const bool last_channel = num_channels == (channel_idx + 1);
  const bool last_pulse   = gridDim.y == (pulse_idx + 1);  // gridDim.y == # pulses
  const bool last_packet  = gridDim.x == (pkt_idx + 1);  // gridDim.x == packets/pulse
  bool set_eoa = last_channel && last_pulse && last_packet;

  // Copy in meta data
  uint16_t *meta = reinterpret_cast<uint16_t *>(pkt);
  meta[0] = sample_idx;
  meta[1] = waveform_id;
  meta[2] = channel_idx;
  meta[3] = pulse_idx;
  meta[4] = pkt_samples;
  meta[5] = set_eoa;

  // Copy over payload
  auto out = reinterpret_cast<complex_t *>(pkt + RFPacket::payload_offset);
  auto in  = &rf_data[sample_idx + num_samples * pulse_idx];
  for (int samp = threadIdx.x; samp < pkt_samples; samp += blockDim.x) {
    out[samp] = in[samp];
  }
}

void AdvConnectorOpTx::populate_packets(uint8_t **out_ptr,
                                        complex_t *rf_data,
                                        uint16_t waveform_id,
                                        uint16_t channel_idx,
                                        uint16_t offset,
                                        cudaStream_t stream) {
  const dim3 grid(pkt_per_pulse, num_pulses_.get(), 1);
  populate_packets_kernel<<<grid, 128, 0, stream>>>(
    out_ptr,
    rf_data,
    waveform_id,
    channel_idx,
    samples_per_pkt,
    pkt_per_pulse,
    num_channels_.get(),
    num_samples_.get(),
    offset);
}

void AdvConnectorOpTx::compute(InputContext& op_input,
                               OutputContext& op_output,
                               ExecutionContext& context) {
  // Check if GPU send is falling behind
  if (gpu_direct_ && (cudaEventQuery(events_[cur_idx]) != cudaSuccess)) {
    HOLOSCAN_LOG_ERROR("Falling behind on TX processing for index {}!", cur_idx);
    return;
  }

  // Input is pulse/sample data from a single channel
  auto rf_data = op_input.receive<std::shared_ptr<RFChannel>>("rf_in").value();
  if (rf_data == nullptr) {
    if (gpu_direct_ && out_q.size() > 0) {
      // If packet setup is done, send to ANO
      const auto first = out_q.front();
      if (cudaEventQuery(first.evt) == cudaSuccess) {
        // Transmit
        HOLOSCAN_LOG_INFO("AdvConnectorOpTx sending {} packets... ({}, {})",
                          adv_net_get_num_pkts(first.msg),
                          first.waveform_id,
                          first.channel_id);
        op_output.emit(first.msg, "burst_out");
        out_q.pop();
      }
    }
    return;
  }
  HOLOSCAN_LOG_INFO("AdvConnectorOpTx::compute()");
  AdvNetStatus ret;

  /**
   * Spin waiting until a buffer is free. This can be stalled by sending
   * faster than the NIC can handle it. We expect the transmit operator to
   * operate much faster than the receiver since it's not having to do any
   * work to construct packets, and just copying from a buffer into memory.
  */
  auto msg = adv_net_create_burst_params();
  adv_net_set_hdr(msg, port_id_.get(), queue_id, num_packets_buf);

  while (!adv_net_tx_burst_available(msg)) {}
  if ((ret = adv_net_get_tx_pkt_burst(msg)) != AdvNetStatus::SUCCESS) {
    HOLOSCAN_LOG_ERROR("Error returned from adv_net_get_tx_pkt_burst: {}",
      static_cast<int>(ret));
    return;
  }

  if (!gpu_direct_) {
    // Generate packets from RF data //todo Optimize this process
    index_t ix_buf = 0;
    index_t ix_max = static_cast<index_t>(num_samples_.get());
    for (index_t ix_pulse = 0; ix_pulse < num_pulses_.get(); ix_pulse++) {
      for (index_t ix_sample = 0; ix_sample < num_samples_.get(); ix_sample += samples_per_pkt) {
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
    if (num_packets_buf != ix_buf || adv_net_get_num_pkts(msg) != ix_buf) {
      HOLOSCAN_LOG_ERROR("Not sending expected number of packets");
    }

    // Send end-of-array message if this is the last channel of the transmit
    const bool send_eoa_msg = rf_data->channel_id == (num_channels_.get() - 1);
    if (send_eoa_msg) {
      packets_buf[num_packets_buf - 1].set_end_array(1);
    }
  }

  // Setup packets
  int cpu_len;
  int gpu_len;
  for (int pkt_idx = 0; pkt_idx < adv_net_get_num_pkts(msg); pkt_idx++) {
    // For HDS mode or CPU mode populate the packet headers
    if (!gpu_direct_ || hds_ > 0) {
      ret = set_cpu_hdr(msg, pkt_idx); // set packet headers
      if (ret != AdvNetStatus::SUCCESS) {
        return;
      }

      // Only set payload on CPU buffer if we're not using GPUDirect
      if (!gpu_direct_) {
        if ((ret = adv_net_set_cpu_udp_payload(msg,
                                               pkt_idx,
                                               packets_buf[pkt_idx].get_ptr(),
                                               payload_size_)) != AdvNetStatus::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set UDP payload for packet {}", pkt_idx);
          adv_net_free_all_burst_pkts_and_burst(msg);
          return;
        }
      }
    }

    // Figure out the CPU and GPU length portions for ANO
    if (gpu_direct_ && hds_ > 0) {
      cpu_len = hds_;
      gpu_len = payload_size_;
    }
    else if (!gpu_direct_) {
      cpu_len = payload_size_ + header_size_.get();  // sizeof UDP header
      gpu_len = 0;
    }
    else {
      cpu_len = 0;
      gpu_len = payload_size_ + header_size_.get();  // sizeof UDP header
    }
    if (gpu_direct_) {
      gpu_bufs[cur_idx][pkt_idx] = reinterpret_cast<uint8_t *>(
        adv_net_get_gpu_pkt_ptr(msg, pkt_idx));
    }

    if ((ret = adv_net_set_pkt_len(msg, pkt_idx, cpu_len, gpu_len)) != AdvNetStatus::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Failed to set lengths for packet {}", pkt_idx);
      adv_net_free_all_burst_pkts_and_burst(msg);
      return;
    }
  }

  // In GPU-only mode copy the header
  if (gpu_direct_ && hds_ == 0) {
    //todo: GPU-only mode
  }

  // Populate packets with I/Q data
  if (gpu_direct_) {
    const auto offset = (hds_ > 0) ? 0 : header_size_.get();
    populate_packets(gpu_bufs[cur_idx],
                     rf_data->data.Data(),
                     rf_data->waveform_id,
                     rf_data->channel_id,
                     offset,
                     streams_[cur_idx]);
    cudaEventRecord(events_[cur_idx], streams_[cur_idx]);
    out_q.push(TxMsg{msg, rf_data->waveform_id, rf_data->channel_id, events_[cur_idx]});
  }

  if (gpu_direct_) {
    // If packet setup is done, send to ANO
    const auto first = out_q.front();
    if (cudaEventQuery(first.evt) == cudaSuccess) {
      // Transmit
      HOLOSCAN_LOG_INFO("AdvConnectorOpTx sending {} packets... ({}, {})",
                        adv_net_get_num_pkts(first.msg),
                        first.waveform_id,
                        first.channel_id);
      op_output.emit(first.msg, "burst_out");
      out_q.pop();
    }
  }
  else {
    // Transmit
    HOLOSCAN_LOG_INFO("AdvConnectorOpTx sending {} packets... ({}, {})",
                      adv_net_get_num_pkts(msg),
                      rf_data->waveform_id,
                      rf_data->channel_id);
    op_output.emit(msg, "burst_out");
  }

  // Increment index
  cur_idx = (++cur_idx % num_concurrent);

  HOLOSCAN_LOG_INFO("AdvConnectorOpTx::compute() done");
}

}  // namespace holoscan::ops
