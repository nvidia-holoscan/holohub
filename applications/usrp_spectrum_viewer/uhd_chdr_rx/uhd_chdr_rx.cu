// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "uhd_chdr_rx.hpp"

#include <cstring>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <utility>

using namespace daqiri;
using namespace matx;

using out_t = std::tuple<tensor_t<complex, 2>, cudaStream_t>;

// Convert one row of CHDR packets into contiguous complex samples.
__global__ void place_packet_data_kernel(complex* out,
                                         const void* const* const __restrict__ in,
                                         const int num_complex_samples_per_packet) {
  const int16_t* samples = reinterpret_cast<const int16_t*>(
      in[(blockIdx.x * blockDim.x) + threadIdx.x]);

  // X4xx CHDR streaming configures the FPGA/link for host-native sc16 layout,
  // so each sample arrives as [real, imaginary] without a software byte swap.
  constexpr float scalar = 1.0F / 32768.0F;
  const size_t offset =
      (static_cast<size_t>(num_complex_samples_per_packet) * blockDim.x * blockIdx.x)
      + (static_cast<size_t>(num_complex_samples_per_packet) * threadIdx.x);

  for (size_t i = 0; i < static_cast<size_t>(num_complex_samples_per_packet); ++i) {
    out[offset + i] = complex(static_cast<float>(samples[i * 2]) * scalar,
                              static_cast<float>(samples[(i * 2) + 1]) * scalar);
  }
}

void place_packet_data(complex* out,
                       const void* const* const in,
                       const int num_outputs_per_batch,
                       const int num_packets_per_output,
                       const int num_complex_samples_per_packet,
                       cudaStream_t stream) {
  place_packet_data_kernel<<<num_outputs_per_batch, num_packets_per_output, 0, stream>>>(
      out, in, num_complex_samples_per_packet);
}

namespace holoscan::ops {

namespace {

void check_cuda(cudaError_t result, const char* operation) {
  if (result != cudaSuccess) {
    throw std::runtime_error(
        fmt::format("{} failed: {}", operation, cudaGetErrorString(result)));
  }
}

void release_bursts(std::vector<BurstParams*>& bursts) {
  for (auto* burst : bursts) {
    if (burst != nullptr) {
      free_all_packets_and_burst_rx(burst);
    }
  }
  bursts.clear();
}

}  // namespace

void UhdChdrRxOp::setup(OperatorSpec& spec) {
  spec.output<out_t>("out");

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
      "Maximum number of converted batches waiting per channel", 2);
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

  if (num_complex_samples_per_packet_.get() == 0
      || num_packets_per_output_.get() == 0
      || num_outputs_per_batch_.get() == 0
      || num_buffered_batches_.get() == 0
      || num_channels_.get() == 0) {
    throw std::runtime_error(
        "uhd_chdr_rx sample, packet, output, buffer, and channel counts must all be > 0");
  }

  int device = 0;
  check_cuda(cudaGetDevice(&device), "cudaGetDevice");
  cudaDeviceProp device_properties{};
  check_cuda(cudaGetDeviceProperties(&device_properties, device), "cudaGetDeviceProperties");
  if (num_packets_per_output_.get() > device_properties.maxThreadsPerBlock) {
    throw std::runtime_error(fmt::format(
        "uhd_chdr_rx.num_packets_per_output ({}) exceeds the CUDA device limit ({})",
        num_packets_per_output_.get(), device_properties.maxThreadsPerBlock));
  }
  if (num_outputs_per_batch_.get() > device_properties.maxGridSize[0]) {
    throw std::runtime_error(fmt::format(
        "uhd_chdr_rx.num_outputs_per_batch ({}) exceeds the CUDA grid limit ({})",
        num_outputs_per_batch_.get(), device_properties.maxGridSize[0]));
  }

  port_id_ = get_port_id(interface_name_.get());
  if (port_id_ == -1) {
    throw std::runtime_error(
        fmt::format("Invalid RX port '{}' specified in the config", interface_name_.get()));
  }

  const auto num_rx_queues = get_num_rx_queues(port_id_);
  if (static_cast<uint64_t>(num_rx_queues) != num_channels_.get()) {
    throw std::runtime_error(fmt::format(
        "uhd_chdr_rx.num_channels ({}) must match the {} DAQIRI RX queue(s) configured "
        "for interface '{}'",
        num_channels_.get(), num_rx_queues, interface_name_.get()));
  }

  num_packets_per_batch_ =
      static_cast<uint64_t>(num_outputs_per_batch_.get()) * num_packets_per_output_.get();

  channel_list.reserve(num_channels_.get());
  try {
    for (uint16_t channel_num = 0; channel_num < num_channels_.get(); ++channel_num) {
      auto channel = std::make_shared<struct Channel>();
      channel->channel_num = channel_num;
      channel->h_dev_ptrs.assign(num_buffered_batches_.get(), nullptr);
      channel->events.assign(num_buffered_batches_.get(), nullptr);
      channel_list.push_back(channel);

      check_cuda(cudaStreamCreateWithFlags(&channel->stream, cudaStreamNonBlocking),
                 "cudaStreamCreateWithFlags");

      for (size_t slot = 0; slot < num_buffered_batches_.get(); ++slot) {
        check_cuda(cudaMallocHost(reinterpret_cast<void**>(&channel->h_dev_ptrs[slot]),
                                  sizeof(void*) * num_packets_per_batch_),
                   "cudaMallocHost");
        check_cuda(cudaEventCreateWithFlags(&channel->events[slot], cudaEventDisableTiming),
                   "cudaEventCreateWithFlags");
      }
    }
  } catch (...) {
    stop();
    throw;
  }
}

std::optional<UhdChdrRxOp::RxMsg> UhdChdrRxOp::free_buf(
    std::shared_ptr<struct Channel> channel) {
  if (channel->out_q.empty()) {
    return std::nullopt;
  }

  const auto query_result = cudaEventQuery(channel->out_q.front().evt);
  if (query_result == cudaErrorNotReady) {
    return std::nullopt;
  }
  check_cuda(query_result, "cudaEventQuery");

  auto completed = std::move(channel->out_q.front());
  channel->out_q.pop();
  release_bursts(completed.bursts);
  return std::optional<RxMsg>{std::in_place, std::move(completed)};
}

bool UhdChdrRxOp::free_bufs_and_emit_arrays(
    OutputContext& op_output,
    std::shared_ptr<struct Channel> channel) {
  const uint16_t channel_num = channel->channel_num;
  auto completed = free_buf(std::move(channel));
  if (!completed.has_value()) {
    return false;
  }

  auto meta = metadata();
  meta->set("channel_number", channel_num);
  op_output.emit(out_t{std::move(completed->data), completed->stream}, "out");
  return true;
}

void UhdChdrRxOp::compute(InputContext&, OutputContext& op_output, ExecutionContext&) {
  const auto num_rx_queues = static_cast<uint16_t>(get_num_rx_queues(port_id_));

  // Emit at most one message per compute call, rotating the starting channel so
  // a continuously ready channel cannot starve the others.
  for (uint16_t i = 0; i < num_rx_queues; ++i) {
    const uint16_t q = (emit_start_ + i) % num_rx_queues;
    if (free_bufs_and_emit_arrays(op_output, channel_list.at(q))) {
      emit_start_ = (q + 1) % num_rx_queues;
      break;
    }
  }

  for (uint16_t q = 0; q < num_rx_queues; ++q) {
    auto channel = channel_list.at(q);
    // Backpressure DAQIRI before starting another aggregate. Continuing a
    // partially filled aggregate is safe because it already owns a queue slot.
    if (channel->aggr_pkts_recv == 0
        && channel->out_q.size() >= num_buffered_batches_.get()) {
      continue;
    }

    BurstParams* burst = nullptr;
    if (get_rx_burst(&burst, port_id_, q) == Status::SUCCESS) {
      process_channel_data(burst, q);
    }
  }
}

void UhdChdrRxOp::discard_current_batch(const std::shared_ptr<struct Channel>& channel) {
  release_bursts(channel->current_bursts);
  channel->aggr_pkts_recv = 0;
}

void UhdChdrRxOp::process_channel_data(BurstParams* burst, uint16_t channel_num) {
  auto channel = channel_list.at(channel_num);
  if (burst == nullptr) {
    throw std::runtime_error("DAQIRI returned a null RX burst");
  }

  const int64_t burst_packets = get_num_packets(burst);
  if (burst_packets <= 0) {
    free_all_packets_and_burst_rx(burst);
    throw std::runtime_error(
        fmt::format("DAQIRI returned an RX burst with {} packets", burst_packets));
  }
  if (channel->aggr_pkts_recv + static_cast<uint64_t>(burst_packets)
      > num_packets_per_batch_) {
    const auto already_aggregated = channel->aggr_pkts_recv;
    discard_current_batch(channel);
    free_all_packets_and_burst_rx(burst);
    throw std::runtime_error(fmt::format(
        "DAQIRI burst ({} packets) would overflow the current batch "
        "({} / {} packets filled); check daqiri.rx.queues[*].batch_size vs "
        "uhd_chdr_rx batching settings",
        burst_packets, already_aggregated, num_packets_per_batch_));
  }

  const uint32_t required_payload_bytes =
      static_cast<uint32_t>(num_complex_samples_per_packet_.get()) * sizeof(uint32_t);
  uint64_t bytes_in_burst = 0;
  auto expected_sequence = channel->next_sequence;
  bool sequence_gap = false;
  uint16_t expected_at_gap = 0;
  uint16_t received_at_gap = 0;

  for (int64_t packet = 0; packet < burst_packets; ++packet) {
    const uint32_t network_header_length = get_segment_packet_length(burst, 0, packet);
    const uint32_t header_length = get_segment_packet_length(burst, 1, packet);
    const uint32_t payload_length = get_segment_packet_length(burst, 2, packet);
    auto* network_header = get_segment_packet_ptr(burst, 0, packet);
    auto* header = get_segment_packet_ptr(burst, 1, packet);
    auto* payload = get_segment_packet_ptr(burst, 2, packet);
    if ((network_header_length != 0 && network_header == nullptr)
        || header_length < sizeof(uint64_t) || header == nullptr
        || payload_length < required_payload_bytes || payload == nullptr) {
      discard_current_batch(channel);
      free_all_packets_and_burst_rx(burst);
      throw std::runtime_error(fmt::format(
          "Invalid CHDR packet on channel {} at burst index {}: network header={} bytes, "
          "CHDR header={} bytes, payload={} bytes (expected CHDR header >= {} and payload >= {})",
          channel_num,
          packet,
          network_header_length,
          header_length,
          payload_length,
          sizeof(uint64_t),
          required_payload_bytes));
    }

    uint64_t header_word = 0;
    std::memcpy(&header_word, header, sizeof(header_word));
    const uint16_t sequence = static_cast<uint16_t>((header_word >> 32) & 0xFFFFU);
    if (expected_sequence.has_value() && sequence != expected_sequence.value()
        && !sequence_gap) {
      sequence_gap = true;
      expected_at_gap = expected_sequence.value();
      received_at_gap = sequence;
    }
    expected_sequence = static_cast<uint16_t>(sequence + 1U);

    const auto destination_index =
        channel->aggr_pkts_recv + static_cast<uint64_t>(packet);
    channel->h_dev_ptrs[channel->cur_idx][destination_index] = payload;
    bytes_in_burst += static_cast<uint64_t>(network_header_length)
        + header_length + payload_length;
  }
  channel->next_sequence = expected_sequence;

  if (sequence_gap) {
    const auto already_aggregated = channel->aggr_pkts_recv;
    discard_current_batch(channel);
    free_all_packets_and_burst_rx(burst);
    HOLOSCAN_LOG_WARN(
        "CHDR sequence gap on channel {}: expected {}, received {}; dropped {} "
        "aggregated packet(s) and the incoming {}-packet burst",
        channel_num,
        expected_at_gap,
        received_at_gap,
        already_aggregated,
        burst_packets);
    return;
  }

  if (log_packets_.get()) {
    HOLOSCAN_LOG_INFO("Processing burst on channel {} (slot {}) with {} packets",
                      channel->channel_num,
                      channel->cur_idx,
                      burst_packets);
    constexpr int packet = 0;
    const uint32_t length0 = get_segment_packet_length(burst, 0, packet);
    const uint32_t length1 = get_segment_packet_length(burst, 1, packet);
    const uint32_t length2 = get_segment_packet_length(burst, 2, packet);
    auto* ptr0 = get_segment_packet_ptr(burst, 0, packet);
    auto* ptr1 = get_segment_packet_ptr(burst, 1, packet);
    auto* ptr2 = get_segment_packet_ptr(burst, 2, packet);
    HOLOSCAN_LOG_INFO("Segment 0 length: {}, Segment 1 length: {}, Segment 2 length: {}",
                      length0, length1, length2);
    HOLOSCAN_LOG_INFO("Segment 0 ptr: {}, Segment 1 ptr: {}, Segment 2 ptr: {}",
                      ptr0, ptr1, ptr2);

    std::ostringstream oss;
    oss << "Segment '0' bytes: ";
    for (uint32_t i = 0; i < length0; ++i) {
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(static_cast<uint8_t*>(ptr0)[i]) << ' ';
    }
    HOLOSCAN_LOG_INFO("{}", oss.str());
    oss.str("");
    oss.clear();
    oss << "Segment '1' bytes: ";
    for (uint32_t i = 0; i < length1; ++i) {
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(static_cast<uint8_t*>(ptr1)[i]) << ' ';
    }
    HOLOSCAN_LOG_INFO("{}", oss.str());

    std::vector<uint8_t> host_buf(length2);
    check_cuda(cudaMemcpy(host_buf.data(), ptr2, length2, cudaMemcpyDeviceToHost),
               "cudaMemcpy");
    oss.str("");
    oss.clear();
    oss << "Segment '2' bytes: ";
    for (const auto value : host_buf) {
      oss << std::hex << std::setw(2) << std::setfill('0')
          << static_cast<int>(value) << ' ';
    }
    HOLOSCAN_LOG_INFO("{}", oss.str());
  }

  channel->ttl_bytes_recv += bytes_in_burst;
  channel->aggr_pkts_recv += static_cast<uint64_t>(burst_packets);
  channel->current_bursts.push_back(burst);

  if (channel->aggr_pkts_recv != num_packets_per_batch_) {
    return;
  }

  HOLOSCAN_LOG_DEBUG("Aggregated {} packets on channel {} slot {} - sending downstream",
                     channel->aggr_pkts_recv,
                     channel->channel_num,
                     channel->cur_idx);

  auto data = make_tensor<complex>(
      {static_cast<index_t>(num_outputs_per_batch_.get()),
       static_cast<index_t>(num_packets_per_output_.get())
           * num_complex_samples_per_packet_.get()},
      MATX_ASYNC_DEVICE_MEMORY,
      channel->stream);
  place_packet_data(data.Data(),
                    channel->h_dev_ptrs[channel->cur_idx],
                    num_outputs_per_batch_.get(),
                    num_packets_per_output_.get(),
                    num_complex_samples_per_packet_.get(),
                    channel->stream);
  check_cuda(cudaGetLastError(), "place_packet_data kernel launch");

  if (log_data_.get()) {
    HOLOSCAN_LOG_INFO("Inspecting RF channel {} data from slot {} with shape: ({}, {})",
                      channel->channel_num,
                      channel->cur_idx,
                      data.Size(0),
                      data.Size(1));
    set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
    print(slice<1>(data, {0, 0}, {matxDropDim, 1024}));
  }

  const auto completed_event = channel->events[channel->cur_idx];
  check_cuda(cudaEventRecord(completed_event, channel->stream), "cudaEventRecord");
  channel->out_q.emplace(std::move(channel->current_bursts),
                         std::move(data),
                         channel->stream,
                         completed_event);
  channel->current_bursts.clear();

  channel->ttl_pkts_recv += channel->aggr_pkts_recv;
  channel->aggr_pkts_recv = 0;
  channel->cur_idx = (channel->cur_idx + 1) % num_buffered_batches_.get();
}

void UhdChdrRxOp::stop() {
  HOLOSCAN_LOG_INFO("UhdChdrRxOp exit report:");
  for (auto& channel : channel_list) {
    HOLOSCAN_LOG_INFO(
        "\n"
        "------- CH {} --------\n"
        "   Processed bytes: {}\n"
        " Processed packets: {}\n",
        channel->channel_num,
        channel->ttl_bytes_recv,
        channel->ttl_pkts_recv);

    if (channel->stream != nullptr) {
      const auto result = cudaStreamSynchronize(channel->stream);
      if (result != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("cudaStreamSynchronize failed on channel {}: {}",
                           channel->channel_num,
                           cudaGetErrorString(result));
      }
    }

    while (!channel->out_q.empty()) {
      release_bursts(channel->out_q.front().bursts);
      channel->out_q.pop();
    }
    discard_current_batch(channel);

    for (auto*& pointers : channel->h_dev_ptrs) {
      if (pointers != nullptr) {
        const auto result = cudaFreeHost(pointers);
        if (result != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("cudaFreeHost failed on channel {}: {}",
                             channel->channel_num,
                             cudaGetErrorString(result));
        }
        pointers = nullptr;
      }
    }
    for (auto& event : channel->events) {
      if (event != nullptr) {
        const auto result = cudaEventDestroy(event);
        if (result != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("cudaEventDestroy failed on channel {}: {}",
                             channel->channel_num,
                             cudaGetErrorString(result));
        }
        event = nullptr;
      }
    }

    // Destroying queued tensor owners can enqueue cudaFreeAsync calls.
    if (channel->stream != nullptr) {
      auto result = cudaStreamSynchronize(channel->stream);
      if (result != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("cudaStreamSynchronize failed on channel {}: {}",
                           channel->channel_num,
                           cudaGetErrorString(result));
      }
      result = cudaStreamDestroy(channel->stream);
      if (result != cudaSuccess) {
        HOLOSCAN_LOG_ERROR("cudaStreamDestroy failed on channel {}: {}",
                           channel->channel_num,
                           cudaGetErrorString(result));
      }
      channel->stream = nullptr;
    }
  }

  channel_list.clear();
}

}  // namespace holoscan::ops
