/*
 * SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
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
#include "vita49_rx.h"
#include "swap.h"
#include "swap.cuh"

using out_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

constexpr uint32_t CONTEXT_QUEUE_ID = 0;

using namespace std::complex_literals;

// CUDA kernel to process an individual VRT packet
__global__ void place_packet_data_kernel(complex* out,
                                         const void* const* const __restrict__ in,
                                         const int cur_idx,
                                         const int num_complex_samples_per_packet
  ) {
  // Warmup
  if (out == nullptr)
    return;

  // The in pointer is an array holding a pointer to the samples for
  // an entire batch (in[640]).
  // blockIdx.x is the packet row and threadIdx.x the packet index
  // This assumes interleaved 16-bit short IQ samples
  const int16_t *samples = reinterpret_cast<const int16_t*>(
          in[(blockIdx.x * blockDim.x) + threadIdx.x]);

  // Scale the int16 values to -1.0 thru +1.0 by dividing by 2^15 - 1 (0x7FFF)
  constexpr float scalar = 1.0 / 0x7FFF;

  // The out pointer is a 4d tensor with structure:
  // 1                        2
  // ---------------------------------------------
  // [P1][P2][P3]...[P20]     [P1][P2][P3]...[P20]
  // [P21][P22]...[P40]       [P21][P22]...[P40]
  // ...                      ...
  // [P12780]...[P12800]      [P12780]...[P12800]
  // We want to get to the index of one of these packets.
  // gridDim.x is num_ffts_per_batch
  // blockDim.x is num_packets_per_fft
  // blockIdx.x is the packet row
  // threadIdx.x is the packet index
  // First, get to the right section of the output tensor (1 or 2),
  // then, index into the row,
  // then, index into the packet
  size_t offset = (num_complex_samples_per_packet * blockDim.x * gridDim.x * cur_idx)
                + (num_complex_samples_per_packet * blockDim.x * blockIdx.x)
                + (num_complex_samples_per_packet * threadIdx.x);

  // Copy data while performing an endian flip and casting to complex float
  for (size_t i = 0; i < num_complex_samples_per_packet; ++i) {
    out[offset + i] = complex(static_cast<float>(bswap_16(samples[i * 2])) * scalar,
                      static_cast<float>(bswap_16(samples[(i * 2) + 1])) * scalar);
  }
}

void place_packet_data(complex* out,
                       const void* const* const in,
                       const uint16_t cur_idx,
                       const int num_ffts_per_batch,
                       const int num_packets_per_fft,
                       const int num_complex_samples_per_packet,
                       cudaStream_t stream) {
  // CUDA execution config <<<Dg, Db, Ns, S>>> where:
  // Dg: dimensionality of the grid of blocks
  // Db: dimensionality of the block of threads
  // Ns: number of bytes in shared memory that is dynamically
  //     allocated _per block_ for this call in addition to
  //     the statically allocated memory
  //  S: associated CUDA stream
  // At this point, we're processing num_ffts_per_batch * num_packets_per_fft packets
  // (e.g. 625 * 20 = 12,500).
  // So, let's launch a grid for every num_packets_per_fft and a thread for every packet.
  // This would make blockIdx.x the packet row and threadIdx.x the packet.
  place_packet_data_kernel<<<
      num_ffts_per_batch,
      num_packets_per_fft,
      num_packets_per_fft * sizeof(int), stream>>>(
          out,
          in,
          cur_idx,
          num_complex_samples_per_packet);
}

namespace holoscan::ops {

void Vita49ConnectorOpRx::setup(OperatorSpec& spec) {
  spec.output<out_t>("out");

  // Data tensor configuration
  // Each packet contains 1024 samples
  // We want to batch up 20 packets for an FFT calculation, so
  // we want 20x1024.
  // We want to perform 625 FFT calculations at once, so we want
  // to batch up 625x20x1024 samples.
  // We want to have 2 data buffers ping-ponging between processing
  // and accumulation, so we want to hold 2x625x20x1024 samples.
  // rf_data is a tensor_t which represents this data. The last
  // dimension (20x1024 in this example) is collapsed into one as
  // downstream operators don't care about how many packets were
  // accumulated.
  spec.param<uint16_t>(num_complex_samples_per_packet_,
      "num_complex_samples_per_packet",
      "Number of complex samples per packet",
      "Number of complex samples per VRT packet", 1024);
  spec.param<uint16_t>(num_packets_per_fft_,
      "num_packets_per_fft",
      "Number of packets per FFT",
      "Number of packets per individual FFT computation", 20);
  spec.param<uint16_t>(num_ffts_per_batch_,
      "num_ffts_per_batch",
      "Number of ffts per batch",
      "Number of fft data batches batches to send for processing at once", 625);
  spec.param<uint16_t>(num_simul_batches_,
      "num_simul_batches",
      "Number of simultaneous batches",
      "Number of simultaneous batches to accumulate/process at once", 2);
  spec.param<uint16_t>(num_channels_,
      "num_channels",
      "Number of channels",
      "Number of channels to process", 2);
  spec.param<std::string>(interface_name_,
      "interface_name",
      "Name of the RX port",
      "Name of the RX port from the advanced_network config",
      "sdr_data");
}

void Vita49ConnectorOpRx::initialize() {
  holoscan::Operator::initialize();

  port_id_ = get_port_id(interface_name_.get());
  if (port_id_ == -1) {
    HOLOSCAN_LOG_ERROR("Invalid RX port {} specified in the config", interface_name_.get());
    exit(1);
  }

  num_packets_per_batch = num_ffts_per_batch_.get() * num_packets_per_fft_.get();

  for (uint16_t channel_num = 0; channel_num < num_channels_.get(); channel_num++) {
    auto new_channel = std::make_shared<struct Channel>();
    new_channel->channel_num = channel_num;
    make_tensor(new_channel->rf_data,
                {num_simul_batches_.get(),
                 num_ffts_per_batch_.get(),
                 num_packets_per_fft_.get() * num_complex_samples_per_packet_.get()});

    // Allocate memory and create CUDA streams for each concurrent batch
    for (int n = 0; n < num_simul_batches_.get(); n++) {
      cudaMallocHost((void**)&new_channel->h_dev_ptrs[n], sizeof(void*) * num_packets_per_batch);

      cudaStreamCreateWithFlags(&new_channel->streams[n], cudaStreamNonBlocking);
      cudaEventCreate(&new_channel->events[n]);
      // Warmup
      place_packet_data(nullptr,
                        nullptr,
                        0,
                        num_ffts_per_batch_.get(),
                        num_packets_per_fft_.get(),
                        num_complex_samples_per_packet_.get(),
                        new_channel->streams[n]);
      cudaStreamSynchronize(new_channel->streams[n]);
    }

    channel_list.push_back(new_channel);
  }
}

std::optional<Vita49ConnectorOpRx::RxMsg> Vita49ConnectorOpRx::free_buf(
        std::shared_ptr<struct Channel> channel) {
  if (!channel->out_q.empty()) {
    auto first = channel->out_q.front();
    if (cudaEventQuery(first.evt) == cudaSuccess) {
      for (auto m = 0; m < first.num_batches; m++) {
        free_all_packets_and_burst_rx(first.msg[m]);
      }
      channel->out_q.pop();
      return std::optional<Vita49ConnectorOpRx::RxMsg>{first};
    }
  }
  return std::nullopt;
}

bool Vita49ConnectorOpRx::free_bufs_and_emit_arrays(
        OutputContext& op_output,
        std::shared_ptr<struct Channel> channel) {
  std::optional<Vita49ConnectorOpRx::RxMsg> completed_msg = free_buf(channel);
  if (!completed_msg.has_value()) {
    return false;
  }

  auto meta = metadata();
  meta->set("channel_number", channel->channel_num);
  meta->set("integer_timestamp", channel->current_meta.integer_time);
  meta->set("fractional_timestamp", channel->current_meta.fractional_time);
  meta->set("stream_id", channel->current_meta.stream_id);
  meta->set("bandwidth_hz", channel->current_context.bandwidth_hz);
  meta->set("rf_ref_freq_hz", channel->current_context.rf_ref_freq_hz);
  meta->set("reference_level_dbm", channel->current_context.reference_level_dbm);
  meta->set("gain_stage_1_db", channel->current_context.gain_stage_1_db);
  meta->set("gain_stage_2_db", channel->current_context.gain_stage_2_db);
  meta->set("sample_rate_hz", channel->current_context.sample_rate_sps);
  if (channel->current_context.context_changed) {
      meta->set("change_indicator", channel->current_context.context_changed);
      channel->current_context.context_changed = false;
  }

  auto data = slice<2>(channel->rf_data, {static_cast<index_t>(channel->cur_idx), 0, 0},
              {matxDropDim, matxEnd, matxEnd});
  op_output.emit(out_t {data, completed_msg.value().stream}, "out");
  return true;
}

void Vita49ConnectorOpRx::compute(
        InputContext& op_input,
        OutputContext& op_output,
        ExecutionContext& context) {
  const auto num_rx_queues = get_num_rx_queues(port_id_);
  // Try to emit any waiting data on any channel that's ready (but
  // only one "emit()" call per "compute()" call).
  for (uint16_t q = CONTEXT_QUEUE_ID + 1; q < num_rx_queues; q++) {
    auto channel = channel_list.at(q - 1);
    if (free_bufs_and_emit_arrays(op_output, channel)) {
      break;
    }
    if (channel->out_q.size() >= num_concurrent) {
      HOLOSCAN_LOG_ERROR("Fell behind in processing on GPU!");
      cudaStreamSynchronize(channel->streams[channel->cur_idx]);
    }
  }

  // Check for a context packet first
  BurstParams *burst;
  auto status = get_rx_burst(&burst, port_id_, CONTEXT_QUEUE_ID);

  // If we have a new context packet, get the metadata out and free
  if (status == Status::SUCCESS) {
    for (int p = 0; p < get_num_packets(burst); p++) {
      // Assume channel 0 context comes in on flow 0, channel 1 on flow 1, etc.
      auto channel_num = get_packet_flow_id(burst, p);

      // Stream ID channel is 1-indexed, but our list is 0-indexed
      if (channel_num >= num_channels_.get()) {
          HOLOSCAN_LOG_CRITICAL("Configured for {} channels, but got context from channel {}",
                                num_channels_.get(), channel_num);
          throw;
      }

      auto channel = channel_list.at(channel_num);

      ContextPacket *ctxt = reinterpret_cast<ContextPacket*>(get_segment_packet_ptr(burst, 1, p));

      // Use lambda here to lazily evaluate the string
      auto log_context_packet = [&]() {
          return fmt::format(
              "Got {}context packet (ch: {}) with:\n"
              "      VRT header: 0x{:X}\n"
              "       Stream ID: 0x{:X}\n"
              "    Integer time: {}\n"
              " Fractional time: {}\n"
              "            CIF0: 0x{:X}\n"
              "       Bandwidth: {} MHz\n"
              "         RF freq: {} MHz\n"
              " Reference level: {} dBm\n"
              "  Gain (stage 1): {} dB\n"
              "  Gain (stage 2): {} dB\n"
              "     Sample rate: {} Msps\n",
              context_changed_h(ctxt) ? "NEW " : "",
              channel_num,
              get_vrt_header_h(&ctxt->metadata),
              get_stream_id_h(&ctxt->metadata),
              get_integer_time_h(&ctxt->metadata),
              get_fractional_time_h(&ctxt->metadata),
              get_cif0_h(ctxt),
              get_bandwidth_hz_h(ctxt) / 1.0e6,
              get_rf_ref_freq_hz_h(ctxt) / 1.0e6,
              get_ref_level_dbm_h(ctxt),
              get_gain_1_db_h(ctxt),
              get_gain_2_db_h(ctxt),
              get_sample_rate_sps_h(ctxt) / 1.0e6);
      };

      if (context_changed_h(ctxt) || !channel->context_received) {
          HOLOSCAN_LOG_INFO("{}", log_context_packet());
      } else {
          HOLOSCAN_LOG_DEBUG("{}", log_context_packet());
      }

      if (!channel->current_context.context_changed) {
          channel->current_context.context_changed = context_changed_h(ctxt);
      }
      channel->current_context.bandwidth_hz = get_bandwidth_hz_h(ctxt);
      channel->current_context.rf_ref_freq_hz = get_rf_ref_freq_hz_h(ctxt);
      channel->current_context.reference_level_dbm = get_ref_level_dbm_h(ctxt);
      channel->current_context.gain_stage_1_db = get_gain_1_db_h(ctxt);
      channel->current_context.gain_stage_2_db = get_gain_2_db_h(ctxt);
      channel->current_context.sample_rate_sps = get_sample_rate_sps_h(ctxt);
      channel->context_received = true;
      // TODO: when context changes, we should flush data
    }
    free_all_packets_and_burst_rx(burst);
    return;
  }

  for (uint16_t q = CONTEXT_QUEUE_ID + 1; q < num_rx_queues; q++) {
    // If there's new data, start processing it
    auto status = get_rx_burst(&burst, port_id_, q);
    if (status == Status::SUCCESS) {
      process_channel_data(op_output, burst, q - 1);
    }
  }
}

void Vita49ConnectorOpRx::process_channel_data(
        OutputContext& op_output,
        BurstParams *burst,
        uint16_t channel_num) {
  auto channel = channel_list.at(channel_num);
  if (!channel->context_received) {
    HOLOSCAN_LOG_INFO("Waiting to process channel {} data until context is received",
                      channel->channel_num);
    free_all_packets_and_burst_rx(burst);
    return;
  }

  // Grab metadata out of the first packet in the batch
  if (!channel->meta_set) {
      VitaMetaData *meta = reinterpret_cast<VitaMetaData*>(get_segment_packet_ptr(burst, 1, 0));
      channel->current_meta.vrt_header = get_vrt_header_h(meta);
      channel->current_meta.stream_id = get_stream_id_h(meta);
      channel->current_meta.integer_time = get_integer_time_h(meta);
      channel->current_meta.fractional_time = get_fractional_time_h(meta);
      channel->meta_set = true;
  }

  uint64_t ttl_bytes_in_cur_batch = 0;
  for (int p = 0; p < get_num_packets(burst); p++) {
      channel->h_dev_ptrs[channel->cur_idx][channel->aggr_pkts_recv + p]
          = get_segment_packet_ptr(burst, 2, p);
      ttl_bytes_in_cur_batch += get_segment_packet_length(burst, 0, p)
          + get_segment_packet_length(burst, 1, p)
          + get_segment_packet_length(burst, 2, p);
  }

  channel->ttl_bytes_recv += ttl_bytes_in_cur_batch;
  channel->aggr_pkts_recv += get_num_packets(burst);
  channel->cur_msg.msg[channel->cur_msg.num_batches++] = burst;

  // Once we've aggregated enough packets, do some work
  if (channel->aggr_pkts_recv >= num_packets_per_batch) {
    HOLOSCAN_LOG_DEBUG("Aggregated {} packets on channel {} index {} - sending downstream",
                      channel->aggr_pkts_recv, channel->channel_num, channel->cur_idx);

    // Copy packet I/Q contents to appropriate location in 'rf_data'
    place_packet_data(channel->rf_data.Data(),
                      channel->h_dev_ptrs[channel->cur_idx],
                      channel->cur_idx,
                      num_ffts_per_batch_.get(),
                      num_packets_per_fft_.get(),
                      num_complex_samples_per_packet_.get(),
                      channel->streams[channel->cur_idx]);

    cudaEventRecord(channel->events[channel->cur_idx], channel->streams[channel->cur_idx]);
    channel->cur_msg.stream = channel->streams[channel->cur_idx];
    channel->cur_msg.evt = channel->events[channel->cur_idx];
    channel->out_q.push(channel->cur_msg);
    channel->cur_msg.num_batches = 0;

    channel->ttl_pkts_recv += channel->aggr_pkts_recv;

    auto ret = cudaGetLastError();
    if (ret != cudaSuccess) {
      HOLOSCAN_LOG_ERROR("CUDA error with {} packets in batch", num_ffts_per_batch_.get());
      HOLOSCAN_LOG_ERROR("Error: {}", cudaGetErrorString(ret));
      exit(1);
    }

    channel->meta_set = false;
    channel->aggr_pkts_recv = 0;
    channel->cur_idx = (channel->cur_idx + 1) % num_simul_batches_.get();
  }
}

void Vita49ConnectorOpRx::stop() {
  HOLOSCAN_LOG_INFO("Vita49ConnectorOpRx exit report:");
  for (uint16_t channel_num = 0; channel_num < num_channels_.get(); channel_num++) {
    auto channel = channel_list.at(channel_num);
    HOLOSCAN_LOG_INFO(
        "\n"
        "------- CH {} --------\n"
        "   Processed bytes: {}\n"
        " Processed packets: {}\n",
        channel->channel_num,
        channel->ttl_bytes_recv,
        channel->ttl_pkts_recv);
  }
}
}  // namespace holoscan::ops
