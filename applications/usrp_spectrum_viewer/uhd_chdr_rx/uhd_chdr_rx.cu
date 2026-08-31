// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "uhd_chdr_rx.hpp"

#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>

using namespace daqiri;
using namespace matx;

using out_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

using namespace std::complex_literals;

// CUDA kernel to process an individual CHDR packet
__global__ void place_packet_data_kernel(complex* out,
                                         const void* const* const __restrict__ in,
                                         const int cur_idx,
                                         const int num_complex_samples_per_packet
  ) {
  // Warmup
  if (out == nullptr)
    return;

  // The in pointer is an array holding one pointer per CHDR packet in a batch,
  // i.e. in[num_outputs_per_batch * num_packets_per_output] (125 * 20 = 2500).
  // blockIdx.x is the packet row and threadIdx.x the packet index
  // This assumes interleaved 16-bit short IQ samples
  const int16_t *samples = reinterpret_cast<const int16_t*>(
          in[(blockIdx.x * blockDim.x) + threadIdx.x]);

  // Scale the int16 values to -1.0 thru +1.0 by dividing by 2^15 - 1 (0x7FFF)
  constexpr float scalar = 1.0 / 0x7FFF;

  // out is a flat complex* into the channel's 3D backing tensor:
  //   [num_buffered_batches][num_outputs_per_batch]
  //   [num_packets_per_output * num_complex_samples_per_packet]
  // The two sections below are the buffered batches; each holds
  // num_outputs_per_batch rows of num_packets_per_output packets
  // (samples within a packet are the flattened innermost dimension):
  // 1                        2
  // ---------------------------------------------
  // [P1][P2][P3]...[P20]     [P1][P2][P3]...[P20]
  // [P21][P22]...[P40]       [P21][P22]...[P40]
  // ...                      ...
  // [P2481]...[P2500]        [P2481]...[P2500]
  // We want to get to the index of one of these packets.
  // gridDim.x is num_outputs_per_batch
  // blockDim.x is num_packets_per_output
  // blockIdx.x is the packet row
  // threadIdx.x is the packet index
  // First, get to the right section of the output tensor (1 or 2),
  // then, index into the row,
  // then, index into the packet
  size_t offset = (num_complex_samples_per_packet * blockDim.x * gridDim.x * cur_idx)
                + (num_complex_samples_per_packet * blockDim.x * blockIdx.x)
                + (num_complex_samples_per_packet * threadIdx.x);

  // Copy the interleaved IQ samples and cast to complex float
  for (size_t i = 0; i < num_complex_samples_per_packet; ++i) {
    out[offset + i] = complex(static_cast<float>(samples[i * 2]) * scalar,
                      static_cast<float>(samples[(i * 2) + 1]) * scalar);
  }
}

void place_packet_data(complex* out,
                       const void* const* const in,
                       const uint16_t cur_idx,
                       const int num_outputs_per_batch,
                       const int num_packets_per_output,
                       const int num_complex_samples_per_packet,
                       cudaStream_t stream) {
  // CUDA execution config <<<Dg, Db, Ns, S>>> where:
  // Dg: dimensionality of the grid of blocks
  // Db: dimensionality of the block of threads
  // Ns: number of bytes in shared memory that is dynamically
  //     allocated _per block_ for this call in addition to
  //     the statically allocated memory
  //  S: associated CUDA stream
  // At this point, we're processing num_outputs_per_batch * num_packets_per_output packets
  // (e.g. 125 * 20 = 2,500).
  // So, let's launch a grid for every num_packets_per_output and a thread for every packet.
  // This would make blockIdx.x the packet row and threadIdx.x the packet.
  place_packet_data_kernel<<<
      num_outputs_per_batch,
      num_packets_per_output,
      0, stream>>>(
          out,
          in,
          cur_idx,
          num_complex_samples_per_packet);
}

namespace holoscan::ops {

void UhdChdrRxOp::setup(OperatorSpec& spec) {
  spec.output<out_t>("out");

  // Data tensor configuration
  // Each packet contains 1024 samples
  // We group 20 packets into one output row, so each row contains 20 x 1024 samples.
  // We emit 125 such rows in one batch, so the batch shape is 125 x (20 x 1024).
  // We keep 2 batches buffered so one can be filled while another is being processed.
  // rf_data stores this as:
  //   [num_buffered_batches][num_outputs_per_batch][num_packets_per_output * samples_per_packet]
  // The packet dimension is flattened because downstream operators consume one contiguous
  // row of samples per output.
  spec.param<uint16_t>(num_complex_samples_per_packet_,
      "num_complex_samples_per_packet",
      "Number of complex samples per packet",
      "Number of complex samples per CHDR packet", 1024);
  spec.param<uint16_t>(num_packets_per_output_,
      "num_packets_per_output",
      "Packets per Output",
      "Number of CHDR packets grouped into one emitted output row", 20);
  spec.param<uint16_t>(num_outputs_per_batch_,
      "num_outputs_per_batch",
      "Outputs per batch",
      "Number of output rows emitted together in one batch", 125);
  spec.param<uint16_t>(num_buffered_batches_,
      "num_buffered_batches",
      "Buffered batches",
      "Number of batch buffers used for concurrent accumulation and processing", 2);
  spec.param<uint16_t>(num_channels_,
      "num_channels",
      "Number of channels",
      "Number of channels to process", 2);
  spec.param<std::string>(interface_name_,
      "interface_name",
      "Name of the RX port",
      "Name of the RX port from the DAQIRI config",
      "sdr_data");
  spec.param<bool>(log_data_,
      "log_data",
      "Log Data",
      "If true, log detailed data information for debugging.", false);
  spec.param<bool>(log_packets_,
      "log_packets",
      "Log Packets",
      "If true, log detailed packet information for debugging.", false);
}

void UhdChdrRxOp::initialize() {
  holoscan::Operator::initialize();

  port_id_ = get_port_id(interface_name_.get());
  if (port_id_ == -1) {
    throw std::runtime_error(
        fmt::format("Invalid RX port '{}' specified in the config", interface_name_.get()));
  }

  // Error checks for num_channels and DAQIRI RX queue matching
  if (const auto num_rx_queues = get_num_rx_queues(port_id_);
      static_cast<uint16_t>(num_rx_queues) != num_channels_.get()) {
    throw std::runtime_error(fmt::format(
        "uhd_chdr_rx.num_channels ({}) must match the {} DAQIRI RX queue(s) configured "
        "for interface '{}'",
        num_channels_.get(), num_rx_queues, interface_name_.get()));
  }

  num_packets_per_batch = num_outputs_per_batch_.get() * num_packets_per_output_.get();

  for (uint16_t channel_num = 0; channel_num < num_channels_.get(); channel_num++) {
    auto new_channel = std::make_shared<struct Channel>();
    new_channel->channel_num = channel_num;
    make_tensor(new_channel->rf_data,
                {num_buffered_batches_.get(),
                 num_outputs_per_batch_.get(),
                 num_packets_per_output_.get() * num_complex_samples_per_packet_.get()});

    // Allocates per-slot buffers/events and one ordered stream per channel.
    // A single stream serializes the whole downstream chain.
    if (num_buffered_batches_.get() == 0 || num_buffered_batches_.get() > num_concurrent) {
      throw std::runtime_error(
          fmt::format("num_buffered_batches must be in [1, {}], got {}",
                      num_concurrent, num_buffered_batches_.get()));
    }

    if (auto err = cudaStreamCreateWithFlags(&new_channel->stream, cudaStreamNonBlocking);
        err != cudaSuccess) {
      throw std::runtime_error(
          fmt::format("cudaStreamCreateWithFlags failed on channel {}: {}",
                      channel_num, cudaGetErrorString(err)));
    }

    for (int n = 0; n < num_buffered_batches_.get(); n++) {
      if (auto err = cudaMallocHost((void**)&new_channel->h_dev_ptrs[n],
                                    sizeof(void*) * num_packets_per_batch);
          err != cudaSuccess) {
        throw std::runtime_error(
            fmt::format("cudaMallocHost failed on channel {} batch {}: {}",
                        channel_num, n, cudaGetErrorString(err)));
      }
      if (auto err = cudaEventCreate(&new_channel->events[n]); err != cudaSuccess) {
        throw std::runtime_error(
            fmt::format("cudaEventCreate failed on channel {} batch {}: {}",
                        channel_num, n, cudaGetErrorString(err)));
      }
      // Warmup
      place_packet_data(nullptr,
                        nullptr,
                        0,
                        num_outputs_per_batch_.get(),
                        num_packets_per_output_.get(),
                        num_complex_samples_per_packet_.get(),
                        new_channel->stream);
      cudaStreamSynchronize(new_channel->stream);
    }

    channel_list.push_back(new_channel);
  }
}

std::optional<UhdChdrRxOp::RxMsg> UhdChdrRxOp::free_buf(
        std::shared_ptr<struct Channel> channel) {
  if (!channel->out_q.empty()) {
    auto first = channel->out_q.front();
    if (cudaEventQuery(first.evt) == cudaSuccess) {
      for (auto m = 0; m < first.num_batches; m++) {
        free_all_packets_and_burst_rx(first.msg[m]);
      }
      channel->out_q.pop();
      return std::optional<UhdChdrRxOp::RxMsg>{first};
    }
  }
  return std::nullopt;
}

bool UhdChdrRxOp::free_bufs_and_emit_arrays(
        OutputContext& op_output,
        std::shared_ptr<struct Channel> channel) {
  std::optional<UhdChdrRxOp::RxMsg> completed_msg = free_buf(channel);
  if (!completed_msg.has_value()) {
    return false;
  }

  auto meta = metadata();
  meta->set("channel_number", channel->channel_num);

  auto data = slice<2>(channel->rf_data,
              {static_cast<index_t>(completed_msg.value().buf_idx), 0, 0},
              {matxDropDim, matxEnd, matxEnd});
  op_output.emit(out_t {data, completed_msg.value().stream}, "out");
  return true;
}

void UhdChdrRxOp::compute(
        InputContext& op_input,
        OutputContext& op_output,
        ExecutionContext& context) {
  const auto num_rx_queues = get_num_rx_queues(port_id_);
  // Try to emit any waiting data on any channel that's ready (but
  // only one "emit()" call per "compute()" call). Start the scan from a
  // rotating channel so a continuously-ready channel cannot starve the others.
  for (uint16_t i = 0; i < num_rx_queues; i++) {
    const uint16_t q = (emit_start_ + i) % num_rx_queues;
    auto channel = channel_list.at(q);
    if (free_bufs_and_emit_arrays(op_output, channel)) {
      emit_start_ = (q + 1) % num_rx_queues;
      break;
    }
    if (channel->out_q.size() >= num_concurrent) {
      HOLOSCAN_LOG_ERROR("Fell behind in processing on GPU!");
      cudaStreamSynchronize(channel->stream);
    }
  }


  BurstParams *burst;
  for (uint16_t q = 0; q < num_rx_queues; q++) {
    // If there's new data, start processing it
    auto status = get_rx_burst(&burst, port_id_, q);
    if (status == Status::SUCCESS) {
      process_channel_data(op_output, burst, q);
    }
  }
}

void UhdChdrRxOp::process_channel_data(
        OutputContext& op_output,
        BurstParams *burst,
        uint16_t channel_num) {
  auto channel = channel_list.at(channel_num);

  // Starting a fresh batch: bound in-flight batches to the physical slot count so
  // a queued message's slot/event is never overwritten before it is emitted.
  if (channel->aggr_pkts_recv == 0) {
    while (channel->out_q.size() >= num_buffered_batches_.get()) {
      auto old = channel->out_q.front();
      cudaEventSynchronize(old.evt);
      for (int m = 0; m < old.num_batches; ++m) {
        free_all_packets_and_burst_rx(old.msg[m]);
      }
      channel->out_q.pop();
      HOLOSCAN_LOG_ERROR("Dropped oldest batch on channel {}: all buffers in flight",
                         channel->channel_num);
    }
  }

  uint64_t ttl_bytes_in_cur_batch = 0;
  const int burst_packets = get_num_packets(burst);
  if (channel->aggr_pkts_recv + static_cast<uint64_t>(burst_packets) > num_packets_per_batch) {
     free_all_packets_and_burst_rx(burst);
     throw std::runtime_error(fmt::format(
         "DAQIRI burst ({} packets) would overflow the current batch ({} / {} packets filled); "
         "check daqiri.rx.queues[*].batch_size vs uhd_chdr_rx batching settings",
         burst_packets,
         channel->aggr_pkts_recv,
         num_packets_per_batch));
  }
  for (int p = 0; p < burst_packets; p++) {
      channel->h_dev_ptrs[channel->cur_idx][channel->aggr_pkts_recv + p]
          = get_segment_packet_ptr(burst, 2, p);
      ttl_bytes_in_cur_batch += get_segment_packet_length(burst, 0, p)
          + get_segment_packet_length(burst, 1, p)
          + get_segment_packet_length(burst, 2, p);
  }

  // Log packet details for debugging
  if (log_packets_) {
    HOLOSCAN_LOG_INFO("Processing burst on channel {} (stream {}) with {} packets",
                    channel->channel_num, channel->cur_idx, get_num_packets(burst));
    int p = 0;
    uint16_t length0 = get_segment_packet_length(burst, 0, p);
    uint16_t length1 = get_segment_packet_length(burst, 1, p);
    uint16_t length2 = get_segment_packet_length(burst, 2, p);
    HOLOSCAN_LOG_INFO("Segment 0 length: {}, Segment 1 length: {}, Segment 2 length: {}",
                      length0, length1, length2);
    auto ptr0 = get_segment_packet_ptr(burst, 0, p);
    auto ptr1 = get_segment_packet_ptr(burst, 1, p);
    auto ptr2 = get_segment_packet_ptr(burst, 2, p);
    HOLOSCAN_LOG_INFO("Segment 0 ptr: {}, Segment 1 ptr: {}, Segment 2 ptr: {}",
                      (void*)ptr0, (void*)ptr1, (void*)ptr2);
    // print bytes for each segment
    std::ostringstream oss;
    oss << "Segment '0' bytes: ";
    for (int i = 0; i < length0; ++i) {
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(((uint8_t*)ptr0)[i]) << ' ';
    }
    HOLOSCAN_LOG_INFO("{}", oss.str());
    oss.str("");
    oss << "Segment '1' bytes: ";
    for (int i = 0; i < length1; ++i) {
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(((uint8_t*)ptr1)[i]) << ' ';
    }
    HOLOSCAN_LOG_INFO("{}", oss.str());
    // copy from device to host memory
    uint8_t* host_buf = nullptr;
    cudaMallocHost((void**)&host_buf, length2);
    cudaMemcpy(host_buf, ptr2, length2, cudaMemcpyDeviceToHost);
    oss.str("");
    oss << "Segment '2' bytes: ";
    for (int i = 0; i < length2; ++i) {
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(host_buf[i]) << ' ';
    }
    HOLOSCAN_LOG_INFO("{}", oss.str());
    cudaFreeHost(host_buf);
  }
  // End packet logging

  channel->ttl_bytes_recv += ttl_bytes_in_cur_batch;
  channel->aggr_pkts_recv += static_cast<uint64_t>(get_num_packets(burst));
  if (channel->cur_msg.num_batches >= MAX_DAQIRI_BATCHES) {
          free_all_packets_and_burst_rx(burst);
          throw std::runtime_error(
                          "Exceeded MAX_DAQIRI_BATCHES while aggregating a batch;"
                          "check daqiri batch_size vs uhd_chdr_rx batching settings");
  }
  channel->cur_msg.msg[channel->cur_msg.num_batches++] = burst;

  // Once we've aggregated enough packets, do some work
  if (channel->aggr_pkts_recv >= num_packets_per_batch) {
    HOLOSCAN_LOG_DEBUG("Aggregated {} packets on channel {} index {} - sending downstream",
                      channel->aggr_pkts_recv, channel->channel_num, channel->cur_idx);

    // Copy packet I/Q contents to appropriate location in 'rf_data'
    place_packet_data(channel->rf_data.Data(),
                      channel->h_dev_ptrs[channel->cur_idx],
                      channel->cur_idx,
                      num_outputs_per_batch_.get(),
                      num_packets_per_output_.get(),
                      num_complex_samples_per_packet_.get(),
                      channel->stream);

    // Log data for debugging
    if (log_data_) {
      HOLOSCAN_LOG_INFO("Inspecting RF channel {} data from thread {} with shape: ({}, {}, {})",
        channel->channel_num, channel->cur_idx,
        channel->rf_data.Size(0), channel->rf_data.Size(1), channel->rf_data.Size(2));
      set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
      print(slice<1>(channel->rf_data,
                     {static_cast<index_t>(channel->cur_idx), 0, 0},
                     {matxDropDim, matxDropDim, 1024}));
    }
    // End data logging

    cudaEventRecord(channel->events[channel->cur_idx], channel->stream);
    channel->cur_msg.stream = channel->stream;
    channel->cur_msg.evt = channel->events[channel->cur_idx];
    channel->cur_msg.buf_idx = channel->cur_idx;
    channel->out_q.push(channel->cur_msg);
    channel->cur_msg.num_batches = 0;

    channel->ttl_pkts_recv += channel->aggr_pkts_recv;

    auto ret = cudaGetLastError();
    if (ret != cudaSuccess) {
      throw std::runtime_error(
          fmt::format("CUDA error with {} packets in batch: {}",
                      num_outputs_per_batch_.get(), cudaGetErrorString(ret)));
    }

    channel->aggr_pkts_recv = 0;
    channel->cur_idx = (channel->cur_idx + 1) % num_buffered_batches_.get();
  }
}

void UhdChdrRxOp::stop() {
  HOLOSCAN_LOG_INFO("UhdChdrRxOp exit report:");
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

     const int batches = (num_buffered_batches_.get() < num_concurrent)
                            ? static_cast<int>(num_buffered_batches_.get())
                            : num_concurrent;
     if (channel->stream != nullptr) {
         cudaStreamSynchronize(channel->stream);
     }

     // Free any outstanding DAQIRI bursts (queued and in-progress).
     while (!channel->out_q.empty()) {
       auto msg = channel->out_q.front();
       for (int m = 0; m < msg.num_batches; ++m) {
         free_all_packets_and_burst_rx(msg.msg[m]);
       }
       channel->out_q.pop();
     }
     for (int m = 0; m < channel->cur_msg.num_batches; ++m) {
       free_all_packets_and_burst_rx(channel->cur_msg.msg[m]);
     }
     channel->cur_msg.num_batches = 0;
     // Release CUDA resources and pinned host allocations.
     for (int n = 0; n < batches; ++n) {
       if (channel->h_dev_ptrs[n] != nullptr) {
         cudaFreeHost(channel->h_dev_ptrs[n]);
         channel->h_dev_ptrs[n] = nullptr;
       }
       if (channel->events[n] != nullptr) {
         cudaEventDestroy(channel->events[n]);
         channel->events[n] = nullptr;
       }
       if (channel->stream != nullptr) {
         cudaStreamDestroy(channel->stream);
         channel->stream = nullptr;
       }
     }
  }

  channel_list.clear();
}
}  // namespace holoscan::ops
