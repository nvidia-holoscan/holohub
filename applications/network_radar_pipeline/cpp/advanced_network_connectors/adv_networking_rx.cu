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
#include "adv_networking_rx.h"  // TODO: Rename networking connectors

#if SPOOF_PACKET_DATA
/**
 * This function converts the packet count to packet metadata. We just treat the
 * packet count as if all of the packets are arriving in order and write the the
 * metadata accordingly. This functionalitycan be useful when testing, where we
 * have a packet generator that isn't generating packets that use our data format.
 */
__device__ __forceinline__
void gen_meta_from_pkt_cnt(RfMetaData *meta,
                           const uint64_t pkt_cnt,
                           const uint16_t num_channels,
                           const uint16_t num_pulses,
                           const uint16_t num_samples,
                           const uint16_t pkts_per_pulse,
                           const uint16_t max_waveform_id) {
  const uint64_t pkts_per_channel  = pkts_per_pulse * num_pulses;
  const uint64_t pkts_per_transmit = pkts_per_channel * num_channels;
  meta->waveform_id = static_cast<uint16_t>((pkt_cnt / pkts_per_transmit) % max_waveform_id);
  meta->channel_idx = static_cast<uint16_t>((pkt_cnt % pkts_per_transmit) / pkts_per_channel);
  meta->pulse_idx   = static_cast<uint16_t>((pkt_cnt % pkts_per_channel) / pkts_per_pulse);
  meta->sample_idx  = static_cast<uint16_t>(SPOOF_SAMPLES_PER_PKT * (pkt_cnt % pkts_per_pulse));
  meta->pkt_samples = min(num_samples - meta->sample_idx, SPOOF_SAMPLES_PER_PKT);
  meta->end_array   = (pkt_cnt + 1) % pkts_per_transmit == 0;
}
#endif

__global__
void place_packet_data_kernel(complex_t *out,
                              const void *const *const __restrict__ in,
                              int *sample_cnt,
                              bool *received_end,
                              const size_t buffer_pos,
                              const uint16_t pkt_len,
                              const uint16_t buffer_size,
                              const uint16_t num_channels,
                              const uint16_t num_pulses,
                              const uint16_t num_samples,
                              const uint64_t total_pkts,
                              const uint16_t pkts_per_pulse,
                              const uint16_t max_waveform_id) {
  const uint32_t channel_stride = static_cast<uint32_t>(num_samples) * num_pulses;
  const uint32_t buffer_stride  = num_channels * channel_stride;
  const uint32_t pkt_idx = blockIdx.x;

#if SPOOF_PACKET_DATA
  // Generate fake packet meta-data from the packet count
  RfMetaData meta_obj;
  RfMetaData *meta = &meta_obj;
  gen_meta_from_pkt_cnt(meta,
                        total_pkts + pkt_idx,
                        num_channels,
                        num_pulses,
                        num_samples,
                        pkts_per_pulse,
                        max_waveform_id);
  const complex_t *samples = reinterpret_cast<const complex_t *>(in[pkt_idx]);
#else
  const RfMetaData *meta   = reinterpret_cast<const RfMetaData *>(in[pkt_idx]);
  const complex_t *samples = reinterpret_cast<const complex_t *>(in[pkt_idx]) + 2;
#endif

  // Make sure this isn't wrapping the buffer - drop if it is
  if (meta->waveform_id >= buffer_pos + buffer_size ||
      meta->waveform_id < buffer_pos) {
    return;
  }

  const uint16_t buffer_idx = meta->waveform_id % buffer_size;
  if (meta->end_array && threadIdx.x == 0) {
    received_end[buffer_idx] = true;
  }

  // Compute pointer in buffer memory
  const uint32_t idx_offset = meta->sample_idx
                            + meta->pulse_idx   * num_samples
                            + meta->channel_idx * channel_stride
                            + buffer_idx        * buffer_stride;

  // Copy data
  for (uint16_t i = threadIdx.x; i < meta->pkt_samples; i += blockDim.x) {
    out[idx_offset + i] = samples[i];
  }

  if (threadIdx.x == 0) {
    // todo Smarter way than atomicAdd
    atomicAdd(&sample_cnt[buffer_idx], meta->pkt_samples);
  }
}

void place_packet_data(complex_t *out,
                       const void *const *const in,
                       int *sample_cnt,
                       bool *received_end,
                       const size_t buffer_pos,
                       const uint16_t pkt_len,
                       const uint32_t num_pkts,
                       const uint16_t buffer_size,
                       const uint16_t num_channels,
                       const uint16_t num_pulses,
                       const uint16_t num_samples,
                       const uint64_t total_pkts,
                       const uint16_t pkts_per_pulse,
                       const uint16_t max_waveform_id,
                       cudaStream_t stream) {
  // Each thread processes an individual packet
  place_packet_data_kernel<<<num_pkts, 128, buffer_size*sizeof(int), stream>>>(
    out,
    in,
    sample_cnt,
    received_end,
    buffer_pos,
    pkt_len,
    buffer_size,
    num_channels,
    num_pulses,
    num_samples,
    total_pkts,
    pkts_per_pulse,
    max_waveform_id);
}

namespace holoscan::ops {

void AdvConnectorOpRx::setup(OperatorSpec& spec) {
  spec.input<std::shared_ptr<AdvNetBurstParams>>("burst_in");
  spec.output<std::shared_ptr<RFArray>>("rf_out");

  // Radar settings
  spec.param<uint16_t>(buffer_size_,
    "buffer_size",
    "Size of RF buffer",
    "Max number of transmits that can be held at once", {});
  spec.param<uint16_t>(num_channels_,
    "num_channels",
    "Number of channels",
    "Number of channels", {});
  spec.param<uint16_t>(num_pulses_,
    "num_pulses",
    "Number of pulses",
    "Number of pulses per channel", {});
  spec.param<uint16_t>(num_samples_,
    "num_samples",
    "Number of samples",
    "Number of samples per pulse", {});

  // Networking settings
  spec.param<bool>(hds_,
    "split_boundary",
    "Header-data split boundary",
    "Byte boundary where header and data is split", true);
  spec.param<bool>(gpu_direct_,
    "gpu_direct",
    "GPUDirect enabled",
    "GPUDirect is enabled for incoming packets", true);
  spec.param<uint32_t>(batch_size_,
    "batch_size",
    "Batch size",
    "Batch size in packets for each processing epoch", 1000);
  spec.param<uint16_t>(max_packet_size_,
    "max_packet_size",
    "Max packet size",
    "Maximum packet size expected from sender", 9100);
  spec.param<uint16_t>(header_size_,
    "header_size",
    "Header size",
    "Header size on each packet from L4 and below", 42);
}

void AdvConnectorOpRx::initialize() {
  HOLOSCAN_LOG_INFO("AdvConnectorOpRx::initialize()");
  holoscan::Operator::initialize();

  cudaStreamCreate(&proc_stream);

  // Assume all packets are the same size, specified in the config
  nom_payload_size_ = max_packet_size_.get() - header_size_.get();

  // Total number of I/Q samples per array
  samples_per_arr = num_channels_.get() * num_pulses_.get() * num_samples_.get();

  // Configuration checks
  if (!(hds_.get() && gpu_direct_.get())) {
    HOLOSCAN_LOG_ERROR("Only configured to run with Header-Data Split and GPUDirect");
    exit(1);
  } else if (hds_.get() && !gpu_direct_.get()) {
    HOLOSCAN_LOG_ERROR("If Header-Data Split mode is enabled, GPUDirect needs to be too");
    exit(1);
  }

  // Allocate memory and create CUDA streams for each concurrent batch
  for (int n = 0; n < num_concurrent; n++) {
    if (gpu_direct_.get()) {
      cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * batch_size_.get());
    }

    buffer_track = AdvBufferTracking(buffer_size_.get());
    make_tensor(rf_data,
      {buffer_size_.get(), num_channels_.get(), num_pulses_.get(), num_samples_.get()});

    cudaStreamCreate(&streams_[n]);
    cudaEventCreate(&events_[n]);
  }

#if SPOOF_PACKET_DATA
  // Compute packets delivered per pulse and max waveform ID based on parameters
  const size_t spoof_pkt_size = sizeof(complex_t) * SPOOF_SAMPLES_PER_PKT + RFPacket::header_size();
  pkts_per_pulse  = static_cast<uint16_t>(packets_per_pulse(spoof_pkt_size, num_samples_.get()));
  max_waveform_id = static_cast<uint16_t>(
    buffer_size_.get() * (65535 / buffer_size_.get()));  // Max of uint16_t
  HOLOSCAN_LOG_WARN("Spoofing packet metadata, ignoring packet header. Pkts / pulse: {}",
    pkts_per_pulse);
  if (spoof_pkt_size >= max_packet_size_.get()) {
    HOLOSCAN_LOG_ERROR("Max packets size ({}) can't fit the expected samples ({})",
      max_packet_size_.get(), SPOOF_SAMPLES_PER_PKT);
    exit(1);
  }
#else
  // These are only used when spoofing packet metadata
  pkts_per_pulse  = 0;
  max_waveform_id = 0;
#endif

  HOLOSCAN_LOG_INFO("AdvConnectorOpRx::initialize() complete");
}

std::vector<AdvConnectorOpRx::RxMsg> AdvConnectorOpRx::free_bufs() {
  std::vector<AdvConnectorOpRx::RxMsg> completed;

  // Loop over all batches, checking if any have completed
  while (out_q.size() > 0) {
    const auto first = out_q.front();
    if (cudaEventQuery(first.evt) == cudaSuccess) {
      completed.push_back(first);
      for (auto m = 0; m < first.num_batches; m++) {
        adv_net_free_all_burst_pkts_and_burst(first.msg[m]);
      }
      out_q.pop();
    } else {
      break;
    }
  }
  return completed;
}

void AdvConnectorOpRx::free_bufs_and_emit_arrays(OutputContext& op_output) {
  std::vector<AdvConnectorOpRx::RxMsg> completed_msgs = free_bufs();
  if (completed_msgs.empty()) {
    return;
  }
  cudaStream_t stream = completed_msgs[0].stream;

  buffer_track.transfer(cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  for (size_t i = 0; i < buffer_track.buffer_size; i++) {
    const size_t pos_wrap = (buffer_track.pos + i) % buffer_track.buffer_size;
    if (!buffer_track.received_end_h[pos_wrap]) {
      continue;
    }

    // Received End-of-Array (EOA) message, emit to downstream operators
    auto params = std::make_shared<RFArray>(
      rf_data->Slice<3>(
        {static_cast<index_t>(pos_wrap), 0, 0, 0},
        {matxDropDim, matxEnd, matxEnd, matxEnd}),
      0, proc_stream);

    op_output.emit(params, "rf_out");
    HOLOSCAN_LOG_INFO("Emitting {} with {}/{} samples",
      buffer_track.pos + i,
      buffer_track.sample_cnt_h[pos_wrap],
      samples_per_arr);

    // Increment the tracker 'i' number of times. This allows us to not get hung on arrays
    // where the EOA was either dropped or missed. Ex: if the EOA for array 11 was dropped,
    // we will emit array 12 when its EOA arrives, incrementing from 10 -> 12.
    for (size_t j = 0; j <= i; j++) {
      buffer_track.increment();
      HOLOSCAN_LOG_INFO("Next waveform expected: {}", buffer_track.pos);
    }

    buffer_track.transfer(cudaMemcpyHostToDevice, stream);
    cudaStreamSynchronize(stream);
    break;
  }
}

void AdvConnectorOpRx::compute(InputContext& op_input,
                               OutputContext& op_output,
                               ExecutionContext& context) {
  // todo Some sort of warm start for the processing stages?
  int64_t ttl_bytes_in_cur_batch_ = 0;

  auto burst_opt = op_input.receive<std::shared_ptr<AdvNetBurstParams>>("burst_in");
  if (!burst_opt) {
    free_bufs();
    return;
  }
  auto burst = burst_opt.value();

  // If packets are coming in from our non-GPUDirect queue, free them and move on
  if (adv_net_get_q_id(burst) == 0) {  // queue 0 is configured to be non-GPUDirect in yaml config
    adv_net_free_cpu_pkts_and_burst(burst);
    HOLOSCAN_LOG_INFO("Freeing CPU packets on queue 0");
    return;
  }

  // Header data split saves off the GPU pointers into a host-pinned buffer to reassemble later.
  // Once enough packets are aggregated, a reorder kernel is launched. In CPU-only mode the
  // entire burst buffer pointer is saved and freed once an entire batch is received.
  if (gpu_direct_.get() && hds_.get()) {
    for (int p = 0; p < adv_net_get_num_pkts(burst); p++) {
      h_dev_ptrs_[cur_idx][aggr_pkts_recv_ + p] = adv_net_get_gpu_pkt_ptr(burst, p);
      ttl_bytes_in_cur_batch_ += adv_net_get_gpu_pkt_len(burst, p)
                               + adv_net_get_cpu_pkt_len(burst, p);
    }
    ttl_bytes_recv_ += ttl_bytes_in_cur_batch_;
  }

  aggr_pkts_recv_ += adv_net_get_num_pkts(burst);
  cur_msg_.msg[cur_msg_.num_batches++] = burst;

  // Once we've aggregated enough packets, do some work
  if (aggr_pkts_recv_ >= batch_size_.get()) {
    if (gpu_direct_.get()) {
      do {
        free_bufs_and_emit_arrays(op_output);
        if (out_q.size() == num_concurrent) {
          HOLOSCAN_LOG_ERROR("Fell behind in processing on GPU!");
          cudaStreamSynchronize(streams_[cur_idx]);
        }
      } while (out_q.size() == num_concurrent);

      // Copy packet I/Q contents to appropriate location in 'rf_data'
      place_packet_data(rf_data->Data(),
                        h_dev_ptrs_[cur_idx],
                        buffer_track.sample_cnt_d,
                        buffer_track.received_end_d,
                        buffer_track.pos,
                        nom_payload_size_,
                        aggr_pkts_recv_,
                        buffer_size_.get(),
                        num_channels_.get(),
                        num_pulses_.get(),
                        num_samples_.get(),
                        ttl_pkts_recv_,   // only needed if spoofing packets
                        pkts_per_pulse,   // only needed if spoofing packets
                        max_waveform_id,  // only needed if spoofing packets
                        streams_[cur_idx]);

      cudaEventRecord(events_[cur_idx], streams_[cur_idx]);
      cur_msg_.stream = streams_[cur_idx];
      cur_msg_.evt    = events_[cur_idx];
      out_q.push(cur_msg_);
      cur_msg_.num_batches = 0;

      ttl_pkts_recv_ += aggr_pkts_recv_;

      if (cudaGetLastError() != cudaSuccess)  {
        HOLOSCAN_LOG_ERROR("CUDA error with {} packets in batch and {} bytes total",
                batch_size_.get(), batch_size_.get()*nom_payload_size_);
        exit(1);
      }
    } else {
      adv_net_free_all_burst_pkts_and_burst(burst);
    }
    aggr_pkts_recv_ = 0;
    cur_idx = (++cur_idx % num_concurrent);
  }
}

void AdvConnectorOpRx::stop() {
  HOLOSCAN_LOG_INFO(
    "\n"
    "AdvConnectorOpRx exit report:\n"
    "--------------------------------\n"
    " - Received bytes:     {}\n"
    " - Received packets:   {}\n",
    ttl_bytes_recv_,
    ttl_pkts_recv_);
}

}  // namespace holoscan::ops
