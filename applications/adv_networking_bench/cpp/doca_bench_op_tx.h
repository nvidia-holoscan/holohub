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
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <arpa/inet.h>
#include <assert.h>
#include <sys/time.h>
#include <unistd.h>

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

class AdvNetworkingBenchDocaTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingBenchDocaTxOp)

  AdvNetworkingBenchDocaTxOp() = default;

  ~AdvNetworkingBenchDocaTxOp() {
    HOLOSCAN_LOG_INFO("ANO benchmark TX op shutting down");
    adv_net_shutdown();
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDocaTxOp::initialize()");
    holoscan::Operator::initialize();

    size_t buf_size = batch_size_.get() * payload_size_.get();

    adv_net_format_eth_addr(eth_dst_, eth_dst_addr_.get());
    inet_pton(AF_INET, ip_src_addr_.get().c_str(), &ip_src_);
    inet_pton(AF_INET, ip_dst_addr_.get().c_str(), &ip_dst_);

    // ANO expects host order when setting
    ip_src_ = ntohl(ip_src_);
    ip_dst_ = ntohl(ip_dst_);

    for (int n = 0; n < num_concurrent; n++) {
      cudaMallocHost(&gpu_bufs[n], sizeof(uint8_t**) * batch_size_.get());
      cudaStreamCreate(&streams_[n]);
      cudaEventCreate(&events_[n]);
      copy_headers(nullptr, nullptr, 0, 0, streams_[n]);
      cudaStreamSynchronize(streams_[n]);
    }

    HOLOSCAN_LOG_INFO("Initialized {} streams and events", num_concurrent);

    // TX GPU-only mode
    // This section simply serves as an example to get an Eth+IP+UDP header onto the GPU,
    // but this header will not be correct without modification of the IP and MAC. In a
    // real situation the header would likely be constructed on the GPU
    if (gpu_direct_.get()) {
      cudaMalloc(&pkt_header_, header_size_.get());

      if ((ip_dst_ & 0xff) == 2) {
        uint8_t payload[] = {0x00, 0x00, 0x00, 0x00, 0x11, 0x22, 0x48, 0xB0, 0x2D, 0xD9, 0x30,
                             0xA1, 0x08, 0x00, 0x45, 0x00, 0x1F, 0x72, 0x00, 0x00, 0x00, 0x00,
                             0x00, 0x11, 0x00, 0x00, 0xC0, 0xA8, 0x00, 0x01, 0xC0, 0xA8, 0x00,
                             0x02, 0x10, 0x00, 0x10, 0x00, 0x1F, 0x5E, 0x00, 0x00, 0x00, 0x00,
                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

        // At this point we have a dummy header created, so we copy it to the GPU
        cudaMemcpy(pkt_header_, payload, sizeof(payload), cudaMemcpyDefault);
      } else {
        uint8_t payload[] = {0x00, 0x00, 0x00, 0x00, 0x11, 0x33, 0x48, 0xB0, 0x2D, 0xD9, 0x30,
                             0xA1, 0x08, 0x00, 0x45, 0x00, 0x1F, 0x72, 0x00, 0x00, 0x00, 0x00,
                             0x00, 0x11, 0x00, 0x00, 0xC0, 0xA8, 0x02, 0x1B, 0xC0, 0xA8, 0x02,
                             0x1C, 0x10, 0x00, 0x10, 0x00, 0x1F, 0x8E, 0x00, 0x00, 0x00, 0x00,
                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                             0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00};

        // At this point we have a dummy header created, so we copy it to the GPU
        cudaMemcpy(pkt_header_, payload, sizeof(payload), cudaMemcpyDefault);
      }
    }

    first_time = true;
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchTxOp::initialize() complete");
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
    spec.param<uint16_t>(port_id_, "port_id", "Interface number", "Interface number");
    spec.param<uint16_t>(header_size_,
                         "header_size",
                         "Header size",
                         "Header size on each packet from L4 and below",
                         42);
    spec.param<bool>(
        gpu_comms_, "gpu_comms", "GPU Communications enabled", "Enable GPU communications", false);
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    AdvNetStatus ret;
    int gpu_len;
    int cpu_len = 0;
    cudaError_t ret_cuda;
    /**
     * Spin waiting until a buffer is free. This can be stalled by sending faster than the NIC can
     * handle it. We expect the transmit operator to operate much faster than the receiver since
     * it's not having to do any work to construct packets, and just copying from a buffer into
     * memory.
     */

    if (!first_time) {
      ret_cuda = cudaEventQuery(events_[cur_idx]);
      if ((ret_cuda != cudaSuccess)) {
        HOLOSCAN_LOG_ERROR(
            "Falling behind on TX processing for index {} ret {}!", cur_idx, ret_cuda);
        return;
      }
    }

    auto msg = adv_net_create_burst_params();
    adv_net_set_hdr(msg, port_id_.get(), queue_id, batch_size_.get());

    // HOLOSCAN_LOG_INFO("Start main thread");

    while (!adv_net_tx_burst_available(msg)) {}
    if ((ret = adv_net_get_tx_pkt_burst(msg)) != AdvNetStatus::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Error returned from adv_net_get_tx_pkt_burst: {}", static_cast<int>(ret));
      return;
    }

    // For HDS mode or CPU mode populate the packet headers
    for (int num_pkt = 0; num_pkt < adv_net_get_num_pkts(msg); num_pkt++) {
      gpu_len = payload_size_.get() + header_size_.get();  // sizeof UDP header
      gpu_bufs[cur_idx][num_pkt] =
          reinterpret_cast<uint8_t*>(adv_net_get_gpu_pkt_ptr(msg, num_pkt));

      if ((ret = adv_net_set_pkt_len(msg, num_pkt, cpu_len, gpu_len)) != AdvNetStatus::SUCCESS) {
        HOLOSCAN_LOG_ERROR("Failed to set lengths for packet {}", num_pkt);
        adv_net_free_all_burst_pkts_and_burst(msg);
        return;
      }
    }

    // In GPU-only mode copy the header
    copy_headers(gpu_bufs[cur_idx],
                 pkt_header_,
                 header_size_.get(),
                 adv_net_get_num_pkts(msg),
                 streams_[cur_idx]);

    // Populate packets with 16-bit numbers of {0,0}, {1,1}, ...
    populate_packets(gpu_bufs[cur_idx],
                     payload_size_.get(),
                     adv_net_get_num_pkts(msg),
                     header_size_.get(),
                     streams_[cur_idx]);
    cudaEventRecord(events_[cur_idx], streams_[cur_idx]);
    msg->event = events_[cur_idx];
    out_q.push(TxMsg{msg, events_[cur_idx]});

    // HOLOSCAN_LOG_INFO("cur_idx {}", cur_idx);
    cur_idx = (++cur_idx % num_concurrent);
    if (first_time && cur_idx == 0) first_time = false;

    const auto first = out_q.front();
    if (cudaEventQuery(first.evt) == cudaSuccess) {
      op_output.emit(first.msg, "burst_out");
      out_q.pop();
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
  uint8_t eth_dst_[6];
  // uint8_t *gpu_bufs[num_concurrent] = {0};
  std::array<uint8_t**, num_concurrent> gpu_bufs;
  uint32_t ip_src_;
  uint32_t ip_dst_;
  cudaStream_t stream;
  void* pkt_header_;
  int cur_idx = 0;
  Parameter<bool> gpu_direct_;  // GPUDirect enabled
  Parameter<bool> gpu_comms_;   // GDAKIN GPU communications enabled
  Parameter<uint32_t> batch_size_;
  Parameter<uint16_t> header_size_;  // Header size of packet
  Parameter<uint16_t> port_id_;
  Parameter<uint16_t> payload_size_;
  Parameter<uint16_t> udp_src_port_;
  Parameter<uint16_t> udp_dst_port_;
  Parameter<std::string> ip_src_addr_;
  Parameter<std::string> ip_dst_addr_;
  Parameter<std::string> eth_dst_addr_;
  bool first_time;
};

}  // namespace holoscan::ops
