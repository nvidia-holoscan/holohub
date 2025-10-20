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

#include "advanced_network/common.h"
#include "advanced_network/kernels.h"
#include "kernels.cuh"
#include "holoscan/holoscan.hpp"
#include <queue>
#include <arpa/inet.h>
#include <assert.h>
#include <sys/time.h>

using namespace holoscan::advanced_network;

namespace holoscan::ops {

/*
  The Advanced Networking Benchmark app uses the Advanced Network library to show how to
  send and receive packets at very high rates. The application is configurable
  to show different scenarios that might be used with the Advanced Network library.
  For both TX and RX, there are three possible modes: CPU-only, Header-data split,
  and GPU-only. CPU-only gives the worst performance of the three, but allows the
  packets to be viewed in CPU memory. Header-data split and GPU-only mode both
  utilize GPUDirect technology to DMA data directly to/from NIC to GPU. GPU-only mode
  may give the highest performance in some cases, but the user must handle the header
  processing on the GPU when using it.

  Both TX and RX show how to do stream pipelining by setting up N CUDA streams on
  launch and pushing work to them asynchronously.
*/

class AdvNetworkingBenchDefaultTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingBenchDefaultTxOp)

  AdvNetworkingBenchDefaultTxOp() = default;

  ~AdvNetworkingBenchDefaultTxOp() {
    HOLOSCAN_LOG_INFO("Advanced Networking Benchmark TX op shutting down");
  }

  void populate_dummy_headers(UDPIPV4Pkt& pkt) {
    // get_mac_addr(port_id_, reinterpret_cast<char*>(&pkt.eth.h_source[0]));
    memcpy(pkt.eth.h_dest, eth_dst_, sizeof(pkt.eth.h_dest));
    pkt.eth.h_proto = htons(0x0800);

    uint16_t ip_len = payload_size_.get() + header_size_.get() - sizeof(pkt.eth);

    pkt.ip.version = 4;
    pkt.ip.daddr = ip_dst_;
    pkt.ip.saddr = ip_src_;
    pkt.ip.ihl = 20 / 4;
    pkt.ip.id = 0;
    pkt.ip.ttl = 2;
    pkt.ip.protocol = IPPROTO_UDP;
    pkt.ip.check = 0;
    pkt.ip.frag_off = 0;
    pkt.ip.tot_len = htons(ip_len);

    pkt.udp.check = 0;
    // Since device-only mode doesn't support updating the GPU memory with different ports we just
    // use the first one here.
    pkt.udp.dest = htons(udp_dst_ports_[0]);
    pkt.udp.source = htons(udp_src_ports_[0]);
    pkt.udp.len = htons(ip_len - sizeof(pkt.ip));
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultTxOp::initialize()");
    holoscan::Operator::initialize();

    port_id_ = get_port_id(interface_name_.get());
    if (port_id_ == -1) {
      HOLOSCAN_LOG_ERROR("Invalid TX port {} specified in the config", interface_name_.get());
      exit(1);
    }

    size_t buf_size = batch_size_.get() * payload_size_.get();
    if (!gpu_direct_.get()) {
      full_batch_data_h_ = malloc(buf_size);
      if (full_batch_data_h_ == nullptr) {
        HOLOSCAN_LOG_ERROR("Failed to allocate CPU batch memory");
        return;
      }

      // Fill in with increasing bytes
      uint8_t* cptr = static_cast<uint8_t*>(full_batch_data_h_);
      uint8_t cur = 0;
      for (int b = 0; b < buf_size; b++) { cptr[b] = cur++; }
    }

    format_eth_addr(eth_dst_, eth_dst_addr_.get());
    inet_pton(AF_INET, ip_src_addr_.get().c_str(), &ip_src_);
    inet_pton(AF_INET, ip_dst_addr_.get().c_str(), &ip_dst_);

    if (gpu_direct_.get()) {
      for (int n = 0; n < num_concurrent; n++) {
        cudaMallocHost(&gpu_bufs[n], sizeof(uint8_t**) * batch_size_.get());
        cudaStreamCreate(&streams_[n]);
        cudaEventCreate(&events_[n]);
      }
      HOLOSCAN_LOG_INFO("Initialized {} streams and events", num_concurrent);
    }

    bool src_port_is_range = udp_src_port_str_.get().find('-') != std::string::npos;
    bool dst_port_is_range = udp_dst_port_str_.get().find('-') != std::string::npos;
    if (src_port_is_range || dst_port_is_range) {
      // If UDP port is a range, we must ensure that either GPUDirect is disabled
      // or Header-Data Split (HDS) is enabled (hds > 0).
      // We make a static L4 header in the packet on the device so it needs to be a constant value.
      if (gpu_direct_.get() && !hds_.get()) {
        HOLOSCAN_LOG_ERROR("Cannot use UDP port range with GPUDirect without HDS");
        exit(1);
      }
    }

    if (src_port_is_range) {
      parse_udp_port_range(udp_src_port_str_.get(), udp_src_ports_);
    } else {
      udp_src_ports_.push_back(static_cast<uint16_t>(std::stoul(udp_src_port_str_.get())));
    }

    if (dst_port_is_range) {
      parse_udp_port_range(udp_dst_port_str_.get(), udp_dst_ports_);
    } else {
      udp_dst_ports_.push_back(static_cast<uint16_t>(std::stoul(udp_dst_port_str_.get())));
    }

    // TX GPU-only mode
    // This section simply serves as an example to get an Eth+IP+UDP header onto the GPU,
    // but this header will not be correct without modification of the IP and MAC. In a
    // real situation the header would likely be constructed on the GPU
    if (gpu_direct_.get() && hds_.get() == 0) {
      cudaMalloc(&gds_header_, header_size_.get());
      cudaMemset(gds_header_, 0, header_size_.get());

      populate_dummy_headers(pkt);
      // advanced_network expects host order when setting
      ip_src_ = ntohl(ip_src_);
      ip_dst_ = ntohl(ip_dst_);

      // Copy the pre-made header to GPU
      cudaMemcpy(gds_header_, reinterpret_cast<void*>(&pkt), sizeof(pkt), cudaMemcpyDefault);
    }

    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultTxOp::initialize() complete");
  }

  void setup(OperatorSpec& spec) override {
    spec.param<uint32_t>(
        batch_size_, "batch_size", "Batch size", "Batch size for each processing epoch", 1000);
    spec.param<uint16_t>(payload_size_,
                         "payload_size",
                         "Payload size",
                         "Payload size to send including HDS portion",
                         1400);
    spec.param<int>(hds_,
                    "split_boundary",
                    "Header-data split boundary",
                    "Byte boundary where header and data is split",
                    0);
    spec.param<bool>(gpu_direct_,
                     "gpu_direct",
                     "GPUDirect enabled",
                     "Byte boundary where header and data is split",
                     false);
    spec.param<std::string>(udp_src_port_str_,
          "udp_src_port", "UDP source port",
          "UDP source port or a range of ports (e.g. 1000-1010)");
    spec.param<std::string>(
        udp_dst_port_str_, "udp_dst_port", "UDP destination port",
        "UDP destination port or a range of ports (e.g. 1000-1010)");
    spec.param<std::string>(ip_src_addr_, "ip_src_addr", "IP source address", "IP source address");
    spec.param<std::string>(
        ip_dst_addr_, "ip_dst_addr", "IP destination address", "IP destination address");
    spec.param<std::string>(eth_dst_addr_,
                            "eth_dst_addr",
                            "Ethernet destination address",
                            "Ethernet destination address");
    spec.param<uint16_t>(header_size_,
                         "header_size",
                         "Header size",
                         "Header size on each packet from L4 and below",
                         42);
    spec.param<std::string>(interface_name_,
                            "interface_name",
                            "Name of NIC from advanced_network config",
                            "Name of NIC from advanced_network config");
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    Status ret;
    static int not_available_count = 0;

    if (gpu_direct_.get() && (cudaEventQuery(events_[cur_idx]) != cudaSuccess)) {
      HOLOSCAN_LOG_ERROR("Falling behind on TX processing for index {}!", cur_idx);
      return;
    }

    auto msg = create_tx_burst_params();
    set_header(msg, port_id_, queue_id, batch_size_.get(), hds_.get() > 0 ? 2 : 1);

    /**
     * Spin waiting until a buffer is free. This can be stalled by sending faster than the NIC can
     * handle it. We expect the transmit operator to operate much faster than the receiver since
     * it's not having to do any work to construct packets, and just copying from a buffer into
     * memory.
     */

    if (!is_tx_burst_available(msg)) {
      if (++not_available_count == 10000) {
        HOLOSCAN_LOG_ERROR(
          "TX port {}, queue {}, burst not available too many times consecutively. "\
          "Make sure memory region has enough buffers",
          port_id_, queue_id);
        not_available_count = 0;
      }
      free_tx_metadata(msg);
      return;
    }

    not_available_count = 0;

    if ((ret = get_tx_packet_burst(msg)) != Status::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Error returned from get_tx_packet_burst: {}", static_cast<int>(ret));
      return;
    }

    // For HDS mode or CPU mode populate the packet headers
    for (int num_pkt = 0; num_pkt < get_num_packets(msg); num_pkt++) {
      if (!gpu_direct_.get() || hds_.get() > 0) {
        if ((ret = set_eth_header(msg, num_pkt, eth_dst_)) != Status::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set Ethernet header for packet {}", num_pkt);
          free_all_packets_and_burst_tx(msg);
          return;
        }

        // Remove Eth + IP size
        const auto ip_len = payload_size_.get() + header_size_.get() - (14 + 20);
        if ((ret = set_ipv4_header(msg, num_pkt, ip_len, 17, ip_src_, ip_dst_)) !=
            Status::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set IP header for packet {}", 0);
          free_all_packets_and_burst_tx(msg);
          return;
        }

        if ((ret = set_udp_header(msg,
                                       num_pkt,
                                       // Remove Eth + IP + UDP headers
                                       payload_size_.get() + header_size_.get() - (14 + 20 + 8),
                                       udp_src_ports_[udp_src_idx_],
                                       udp_dst_ports_[udp_dst_idx_])) != Status::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set UDP header for packet {}", 0);
          free_all_packets_and_burst_tx(msg);
          return;
        }

        udp_src_idx_ = (++udp_src_idx_ % udp_src_ports_.size());
        udp_dst_idx_ = (++udp_dst_idx_ % udp_dst_ports_.size());

        // Only set payload on CPU buffer if we're not in HDS mode
        if (hds_.get() == 0) {
          if ((ret = set_udp_payload(
                   msg,
                   num_pkt,
                   static_cast<char*>(full_batch_data_h_) + num_pkt * payload_size_.get(),
                   payload_size_.get())) != Status::SUCCESS) {
            HOLOSCAN_LOG_ERROR("Failed to set UDP payload for packet {}", num_pkt);
            free_all_packets_and_burst_tx(msg);
            return;
          }
        }
      }

      // Figure out the CPU and GPU length portions for advanced_network
      if (gpu_direct_.get() && hds_.get() > 0) {
        gpu_bufs[cur_idx][num_pkt] =
            reinterpret_cast<uint8_t*>(get_segment_packet_ptr(msg, 1, num_pkt));
        if ((ret = set_packet_lengths(msg, num_pkt, {hds_.get(), payload_size_.get()})) !=
            Status::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set lengths for packet {}", num_pkt);
          free_all_packets_and_burst_tx(msg);
          return;
        }
      } else {
        if (gpu_direct_.get()) {
          gpu_bufs[cur_idx][num_pkt] =
              reinterpret_cast<uint8_t*>(get_segment_packet_ptr(msg, 0, num_pkt));
        }

        if ((ret =
                 set_packet_lengths(msg, num_pkt, {payload_size_.get() + header_size_.get()})) !=
            Status::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set lengths for packet {}", num_pkt);
          free_all_packets_and_burst_tx(msg);
          return;
        }
      }
    }

    // In GPU-only mode copy the header
    if (gpu_direct_.get() && hds_.get() == 0) {
      copy_headers(gpu_bufs[cur_idx],
                   gds_header_,
                   header_size_.get(),
                   get_num_packets(msg),
                   streams_[cur_idx]);
    }

    // Populate packets with 16-bit numbers of {0,0}, {1,1}, ...
    if (gpu_direct_.get()) {
      const auto offset = (hds_.get() > 0) ? 0 : header_size_.get();
      populate_packets(gpu_bufs[cur_idx],
                       payload_size_.get(),
                       get_num_packets(msg),
                       offset,
                       streams_[cur_idx]);
      cudaEventRecord(events_[cur_idx], streams_[cur_idx]);
      out_q.push(TxMsg{msg, events_[cur_idx]});
    }

    cur_idx = (++cur_idx % num_concurrent);

    if (gpu_direct_.get()) {
      const auto first = out_q.front();
      if (cudaEventQuery(first.evt) == cudaSuccess) {
        send_tx_burst(first.msg);
        out_q.pop();
      }
    } else {
      send_tx_burst(msg);
    }
  };

 private:
  struct TxMsg {
    BurstParams* msg;
    cudaEvent_t evt;
  };

  static constexpr int num_concurrent = 4;
  std::queue<TxMsg> out_q;
  std::array<cudaStream_t, num_concurrent> streams_;
  std::array<cudaEvent_t, num_concurrent> events_;
  void* full_batch_data_h_;
  static constexpr uint16_t queue_id = 0;
  char eth_dst_[6];
  std::array<uint8_t**, num_concurrent> gpu_bufs;
  uint32_t ip_src_;
  uint32_t ip_dst_;
  cudaStream_t stream;
  UDPIPV4Pkt pkt;
  void* gds_header_;
  int cur_idx = 0;
  int port_id_;
  size_t udp_src_idx_ = 0;
  size_t udp_dst_idx_ = 0;
  std::vector<uint16_t> udp_src_ports_;
  std::vector<uint16_t> udp_dst_ports_;
  Parameter<int> hds_;          // Header-data split point
  Parameter<bool> gpu_direct_;  // GPUDirect enabled
  Parameter<uint32_t> batch_size_;
  Parameter<uint16_t> header_size_;  // Header size of packet
  Parameter<std::string> interface_name_;
  Parameter<uint16_t> payload_size_;
  Parameter<std::string> udp_src_port_str_;
  Parameter<std::string> udp_dst_port_str_;
  Parameter<std::string> ip_src_addr_;
  Parameter<std::string> ip_dst_addr_;
  Parameter<std::string> eth_dst_addr_;

  // Private helper function to parse port ranges
  void parse_udp_port_range(const std::string& port_str, std::vector<uint16_t>& ports_vec) {
    ports_vec.clear();
    size_t dash_pos = port_str.find('-');
    if (dash_pos == std::string::npos) {
      // Single port
      try {
        uint16_t port = static_cast<uint16_t>(std::stoul(port_str));
        ports_vec.push_back(port);
      } catch (const std::invalid_argument& ia) {
        HOLOSCAN_LOG_ERROR("Invalid port format: {}", port_str);
      } catch (const std::out_of_range& oor) {
        HOLOSCAN_LOG_ERROR("Port out of range: {}", port_str);
      }
    } else {
      // Port range
      std::string start_str = port_str.substr(0, dash_pos);
      std::string end_str = port_str.substr(dash_pos + 1);
      try {
        uint16_t start_port = static_cast<uint16_t>(std::stoul(start_str));
        uint16_t end_port = static_cast<uint16_t>(std::stoul(end_str));

        if (start_port > end_port) {
          HOLOSCAN_LOG_ERROR("Invalid port range: start port {} > end port {}",
              start_port, end_port);
          return;
        }

        for (uint16_t port = start_port; port <= end_port; ++port) {
          ports_vec.push_back(port);
          // Check for overflow before incrementing in the loop condition itself if end_port
          // is 65535
          if (port == 65535 && port < end_port) {
            break;
          }
        }
      } catch (const std::invalid_argument& ia) {
        HOLOSCAN_LOG_ERROR("Invalid port format in range: {}", port_str);
      } catch (const std::out_of_range& oor) {
        HOLOSCAN_LOG_ERROR("Port out of range in range: {}", port_str);
      }
    }
  }
};

}  // namespace holoscan::ops
