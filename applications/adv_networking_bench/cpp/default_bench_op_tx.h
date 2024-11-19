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

#include "adv_network_tx.h"
#include "adv_network_kernels.h"
#include "kernels.cuh"
#include "holoscan/holoscan.hpp"
#include <queue>
#include <arpa/inet.h>
#include <assert.h>
#include <sys/time.h>

namespace holoscan::ops {

/*
  The ANO benchmark app uses the Advanced Networking Operator to show how to send
  and receive packets at very high rates. The application is highly configurable
  to show different scenarios that might be used with the ANO. For both TX and RX,
  there are three possible modes: CPU-only, Header-data split, and GPU-only. CPU-only
  gives the worst performance of the three, but allows the packets to be viewed
  in CPU memory. Header-data split and GPU-only mode both utilize GPUDirect technology
  to DMA data directly to/from NIC to GPU. GPU-only mode may give the highest
  performance in some cases, but the user must handle the header processing on the
  GPU when using it.

  Both TX and RX show how to do stream pipelining by setting up N CUDA streams on
  launch and pushing work to them asynchronously.
*/

class AdvNetworkingBenchDefaultTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingBenchDefaultTxOp)

  AdvNetworkingBenchDefaultTxOp() = default;

  ~AdvNetworkingBenchDefaultTxOp() {
    HOLOSCAN_LOG_INFO("ANO benchmark TX op shutting down");
  }

  void populate_dummy_headers(UDPIPV4Pkt& pkt) {
    // adv_net_get_mac(port_id_, reinterpret_cast<char*>(&pkt.eth.h_source[0]));
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
    pkt.udp.dest = htons(udp_dst_port_.get());
    pkt.udp.source = htons(udp_src_port_.get());
    pkt.udp.len = htons(ip_len - sizeof(pkt.ip));
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultTxOp::initialize()");
    holoscan::Operator::initialize();

    // port_id_ = adv_net_address_to_port(address_.get());
    port_id_ = 0;

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

    adv_net_format_eth_addr(eth_dst_, eth_dst_addr_.get());
    inet_pton(AF_INET, ip_src_addr_.get().c_str(), &ip_src_);
    inet_pton(AF_INET, ip_dst_addr_.get().c_str(), &ip_dst_);

    // ANO expects host order when setting
    ip_src_ = ntohl(ip_src_);
    ip_dst_ = ntohl(ip_dst_);

    if (gpu_direct_.get()) {
      for (int n = 0; n < num_concurrent; n++) {
        cudaMallocHost(&gpu_bufs[n], sizeof(uint8_t**) * batch_size_.get());
        cudaStreamCreate(&streams_[n]);
        cudaEventCreate(&events_[n]);
      }
      HOLOSCAN_LOG_INFO("Initialized {} streams and events", num_concurrent);
    }

    // TX GPU-only mode
    // This section simply serves as an example to get an Eth+IP+UDP header onto the GPU,
    // but this header will not be correct without modification of the IP and MAC. In a
    // real situation the header would likely be constructed on the GPU
    if (gpu_direct_.get() && hds_.get() == 0) {
      cudaMalloc(&gds_header_, header_size_.get());
      cudaMemset(gds_header_, 0, header_size_.get());

      populate_dummy_headers(pkt);

      // Copy the pre-made header to GPU
      cudaMemcpy(gds_header_, reinterpret_cast<void*>(&pkt), sizeof(pkt), cudaMemcpyDefault);
    }

    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultTxOp::initialize() complete");
  }

  void setup(OperatorSpec& spec) override {
    spec.output<std::shared_ptr<AdvNetBurstParams>>("burst_out");

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
    spec.param<uint16_t>(udp_src_port_, "udp_src_port", "UDP source port", "UDP source port");
    spec.param<uint16_t>(
        udp_dst_port_, "udp_dst_port", "UDP destination port", "UDP destination port");
    spec.param<std::string>(ip_src_addr_, "ip_src_addr", "IP source address", "IP source address");
    spec.param<std::string>(
        ip_dst_addr_, "ip_dst_addr", "IP destination address", "IP destination address");
    spec.param<std::string>(eth_dst_addr_,
                            "eth_dst_addr",
                            "Ethernet destination address",
                            "Ethernet destination address");
    spec.param<std::string>(
        address_, "address", "Address of NIC from ANO config", "Address of NIC from ANO config");
    spec.param<uint16_t>(header_size_,
                         "header_size",
                         "Header size",
                         "Header size on each packet from L4 and below",
                         42);
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    AdvNetStatus ret;
    /**
     * Spin waiting until a buffer is free. This can be stalled by sending faster than the NIC can
     * handle it. We expect the transmit operator to operate much faster than the receiver since
     * it's not having to do any work to construct packets, and just copying from a buffer into
     * memory.
     */

    if (gpu_direct_.get() && (cudaEventQuery(events_[cur_idx]) != cudaSuccess)) {
      HOLOSCAN_LOG_ERROR("Falling behind on TX processing for index {}!", cur_idx);
      return;
    }

    auto msg = adv_net_create_burst_params();
    adv_net_set_hdr(msg, port_id_, queue_id, batch_size_.get(), hds_.get() > 0 ? 2 : 1);

    while (!adv_net_tx_burst_available(msg)) {}
    if ((ret = adv_net_get_tx_pkt_burst(msg)) != AdvNetStatus::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Error returned from adv_net_get_tx_pkt_burst: {}", static_cast<int>(ret));
      return;
    }

    // For HDS mode or CPU mode populate the packet headers
    for (int num_pkt = 0; num_pkt < adv_net_get_num_pkts(msg); num_pkt++) {
      if (!gpu_direct_.get() || hds_.get() > 0) {
        if ((ret = adv_net_set_eth_hdr(msg, num_pkt, eth_dst_)) != AdvNetStatus::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set Ethernet header for packet {}", num_pkt);
          adv_net_free_all_pkts_and_burst(msg);
          return;
        }

        // Remove Eth + IP size
        const auto ip_len = payload_size_.get() + header_size_.get() - (14 + 20);
        if ((ret = adv_net_set_ipv4_hdr(msg, num_pkt, ip_len, 17, ip_src_, ip_dst_)) !=
            AdvNetStatus::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set IP header for packet {}", 0);
          adv_net_free_all_pkts_and_burst(msg);
          return;
        }

        if ((ret = adv_net_set_udp_hdr(msg,
                                       num_pkt,
                                       // Remove Eth + IP + UDP headers
                                       payload_size_.get() + header_size_.get() - (14 + 20 + 8),
                                       udp_src_port_.get(),
                                       udp_dst_port_.get())) != AdvNetStatus::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set UDP header for packet {}", 0);
          adv_net_free_all_pkts_and_burst(msg);
          return;
        }

        // Only set payload on CPU buffer if we're not in HDS mode
        if (hds_.get() == 0) {
          if ((ret = adv_net_set_udp_payload(
                   msg,
                   num_pkt,
                   static_cast<char*>(full_batch_data_h_) + num_pkt * payload_size_.get(),
                   payload_size_.get())) != AdvNetStatus::SUCCESS) {
            HOLOSCAN_LOG_ERROR("Failed to set UDP payload for packet {}", num_pkt);
            adv_net_free_all_pkts_and_burst(msg);
            return;
          }
        }
      }

      // Figure out the CPU and GPU length portions for ANO
      if (gpu_direct_.get() && hds_.get() > 0) {
        gpu_bufs[cur_idx][num_pkt] =
            reinterpret_cast<uint8_t*>(adv_net_get_seg_pkt_ptr(msg, 1, num_pkt));
        if ((ret = adv_net_set_pkt_lens(msg, num_pkt, {hds_.get(), payload_size_.get()})) !=
            AdvNetStatus::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set lengths for packet {}", num_pkt);
          adv_net_free_all_pkts_and_burst(msg);
          return;
        }
      } else {
        if (gpu_direct_.get()) {
          gpu_bufs[cur_idx][num_pkt] =
              reinterpret_cast<uint8_t*>(adv_net_get_seg_pkt_ptr(msg, 0, num_pkt));
        }

        if ((ret =
                 adv_net_set_pkt_lens(msg, num_pkt, {payload_size_.get() + header_size_.get()})) !=
            AdvNetStatus::SUCCESS) {
          HOLOSCAN_LOG_ERROR("Failed to set lengths for packet {}", num_pkt);
          adv_net_free_all_pkts_and_burst(msg);
          return;
        }
      }
    }

    // In GPU-only mode copy the header
    if (gpu_direct_.get() && hds_.get() == 0) {
      copy_headers(gpu_bufs[cur_idx],
                   gds_header_,
                   header_size_.get(),
                   adv_net_get_num_pkts(msg),
                   streams_[cur_idx]);
    }

    // Populate packets with 16-bit numbers of {0,0}, {1,1}, ...
    if (gpu_direct_.get()) {
      const auto offset = (hds_.get() > 0) ? 0 : header_size_.get();
      populate_packets(gpu_bufs[cur_idx],
                       payload_size_.get(),
                       adv_net_get_num_pkts(msg),
                       offset,
                       streams_[cur_idx]);
      cudaEventRecord(events_[cur_idx], streams_[cur_idx]);
      out_q.push(TxMsg{msg, events_[cur_idx]});
    }

    cur_idx = (++cur_idx % num_concurrent);

    if (gpu_direct_.get()) {
      const auto first = out_q.front();
      if (cudaEventQuery(first.evt) == cudaSuccess) {
        op_output.emit(first.msg, "burst_out");
        out_q.pop();
      }
    } else {
      op_output.emit(msg, "burst_out");
    }
  };

 private:
  struct TxMsg {
    AdvNetBurstParams* msg;
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
  Parameter<int> hds_;          // Header-data split point
  Parameter<bool> gpu_direct_;  // GPUDirect enabled
  Parameter<uint32_t> batch_size_;
  Parameter<uint16_t> header_size_;  // Header size of packet
  Parameter<std::string> address_;
  Parameter<uint16_t> payload_size_;
  Parameter<uint16_t> udp_src_port_;
  Parameter<uint16_t> udp_dst_port_;
  Parameter<std::string> ip_src_addr_;
  Parameter<std::string> ip_dst_addr_;
  Parameter<std::string> eth_dst_addr_;
};

}  // namespace holoscan::ops
