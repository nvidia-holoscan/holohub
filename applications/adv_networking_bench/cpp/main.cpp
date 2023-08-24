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

#include "adv_network_rx.h"
#include "adv_network_tx.h"
#include "adv_network_kernels.h"
#include "holoscan/holoscan.hpp"
#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <arpa/inet.h>
#include <assert.h>


namespace holoscan::ops {

// Example IPV4 UDP packet using Linux headers
struct UDPIPV4Pkt {
  struct ethhdr eth;
  struct iphdr ip;
  struct udphdr udp;
  uint8_t payload[];
} __attribute__((packed));

class AdvNetworkingBenchTxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingBenchTxOp)

  AdvNetworkingBenchTxOp() = default;

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchTxOp::initialize()");
    holoscan::Operator::initialize();

    size_t buf_size = batch_size_.get() * payload_size_.get();
    cudaMallocHost(&full_batch_data_h_, buf_size);

    // Fill in with increasing bytes
    uint8_t *cptr = static_cast<uint8_t*>(full_batch_data_h_);
    uint8_t cur = 0;
    for (int b = 0; b < buf_size; b++) {
      cptr[b] = cur++;
    }

    HOLOSCAN_LOG_INFO("AdvNetworkingBenchTxOp::initialize() complete");
  }

  void setup(OperatorSpec& spec) override {
    spec.output<std::shared_ptr<AdvNetBurstParams>>("burst_out");

    spec.param<uint32_t>(batch_size_, "batch_size", "Batch size",
      "Batch size for each processing epoch", 1000);
    spec.param<uint16_t>(payload_size_, "payload_size", "Payload size",
      "Payload size to send. Does not include <= L4 headers", 1400);
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    AdvNetStatus ret;

    /**
     * Spin waiting until a buffer is free. This can be stalled by sending faster than the NIC can handle it. We
     * expect the transmit operator to operate much faster than the receiver since it's not having to do any work
     * to construct packets, and just copying from a buffer into memory.
    */
    while (!adv_net_tx_burst_available(batch_size_.get())) {}

    auto msg = adv_net_create_shared_burst_params();
    adv_net_set_hdr(msg, port_id, queue_id, batch_size_);

    if ((ret = adv_net_get_tx_pkt_burst(msg)) != AdvNetStatus::SUCCESS) {
      HOLOSCAN_LOG_ERROR("Error returned from adv_net_get_tx_pkt_burst: {}", static_cast<int>(ret));
      return;
    }

    void *pkt;
    for (int num_pkt = 0; num_pkt < adv_net_get_num_pkts(msg); num_pkt++) {
      if ((ret = adv_net_set_cpu_udp_payload( msg,
                                              num_pkt,
                                              static_cast<char*>(full_batch_data_h_) +
                                                    num_pkt * payload_size_.get(),
                                              payload_size_.get())) != AdvNetStatus::SUCCESS) {
        HOLOSCAN_LOG_ERROR("Failed to create packet {}", num_pkt);
      }
    }

    op_output.emit(msg, "burst_out");
  };

 private:
  void *full_batch_data_h_;
  static constexpr uint16_t port_id = 0;
  static constexpr uint16_t queue_id = 0;
  Parameter<uint32_t> batch_size_;
  Parameter<uint16_t> payload_size_;
};

class AdvNetworkingBenchRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingBenchRxOp)

  AdvNetworkingBenchRxOp() = default;

  ~AdvNetworkingBenchRxOp() {
    HOLOSCAN_LOG_INFO("Finished receiver with {}/{} bytes/packets received",
        ttl_bytes_recv_, ttl_pkts_recv_);
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchRxOp::initialize()");
    holoscan::Operator::initialize();

    // For this example assume all packets are the same size, specified in the config
    nom_payload_size_ = max_packet_size_.get() - sizeof(UDPIPV4Pkt);

    cudaMallocHost(&full_batch_data_h_, batch_size_.get() * nom_payload_size_);
    cudaMalloc(&full_batch_data_d_,     batch_size_.get() * nom_payload_size_);

    if (hds_.get()) {
      cudaMallocHost((void**)&h_dev_ptrs_, sizeof(void*) * batch_size_.get());
    }

    HOLOSCAN_LOG_INFO("AdvNetworkingBenchRxOp::initialize() complete");
  }

  void setup(OperatorSpec& spec) override {
    spec.input<AdvNetBurstParams>("burst_in");
    spec.param<bool>(hds_, "split_boundary", "Header-data split boundary",
        "Byte boundary where header and data is split", false);
    spec.param<uint32_t>(batch_size_, "batch_size", "Batch size",
        "Batch size in packets for each processing epoch", 1000);
    spec.param<uint16_t>(max_packet_size_, "max_packet_size",
        "Max packet size", "Maximum packet size expected from sender", 9100);
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {
    int64_t ttl_bytes_in_cur_batch_   = 0;
    auto burst = op_input.receive<std::shared_ptr<AdvNetBurstParams>>("burst_in").value();
    ttl_pkts_recv_ += adv_net_get_num_pkts(burst);

    // If packets are coming in from our non-GPUDirect queue, free them and move on
    if (adv_net_get_q_id(burst) == 0) {
      adv_net_free_cpu_pkts_and_burst(burst);
      HOLOSCAN_LOG_INFO("Freeing CPU packets on queue 0");
      return;
    }

    /* Header data split saves off the GPU pointers into a host-pinned buffer to reassemble later.
     * Once enough packets are aggregated, a reorder kernel is launched. In CPU-only mode the
     * entire burst buffer pointer is saved and freed once an entire batch is received.
     */
    if (hds_.get()) {
      int64_t bytes_in_batch = 0;
      for (int p = 0; p < adv_net_get_num_pkts(burst); p++) {
        h_dev_ptrs_[aggr_pkts_recv_ + p]   = adv_net_get_gpu_pkt_ptr(burst, p);
        ttl_bytes_in_cur_batch_           +=
          adv_net_get_gpu_packet_len(burst, p) + sizeof(UDPIPV4Pkt);
      }

      ttl_bytes_recv_ += ttl_bytes_in_cur_batch_;
    } else {
      auto batch_offset =  aggr_pkts_recv_ * nom_payload_size_;
      for (int p = 0; p < adv_net_get_num_pkts(burst); p++) {
        auto pkt = static_cast<UDPIPV4Pkt*>(adv_net_get_cpu_pkt_ptr(burst, p));
        auto len = ntohs(pkt->udp.len) - 8;

        // assert(len + sizeof(UDPIPV4Pkt) == max_packet_size_.get());

        memcpy((char*)full_batch_data_h_ + batch_offset + p * nom_payload_size_,
            pkt->payload, len);

        ttl_bytes_recv_ += len + sizeof(UDPIPV4Pkt);
        ttl_bytes_in_cur_batch_ += len + sizeof(UDPIPV4Pkt);
      }
    }

    burst_bufs_[burst_buf_idx_++] = burst;
    aggr_pkts_recv_ += adv_net_get_num_pkts(burst);

    if (aggr_pkts_recv_ >= batch_size_.get()) {
      // Do some work on full_batch_data_h_ or full_batch_data_d_
      aggr_pkts_recv_ = 0;

      if (hds_.get()) {
        simple_packet_reorder(static_cast<uint8_t*>(full_batch_data_d_), h_dev_ptrs_,
                      nom_payload_size_, batch_size_.get());
        if (cudaGetLastError() != cudaSuccess)  {
          HOLOSCAN_LOG_ERROR("CUDA error with {} packets in batch and {} bytes total",
                  batch_size_.get(), batch_size_.get()*nom_payload_size_);
          exit(1);
        }

        for (int b = 0; b < burst_buf_idx_; b++) {
          adv_net_free_all_burst_pkts_and_burst(burst_bufs_[b]);
        }
      } else {
        for (int b = 0; b < burst_buf_idx_; b++) {
          adv_net_free_cpu_pkts_and_burst(burst_bufs_[b]);
        }
      }

      burst_buf_idx_ = 0;
    }
  }

 private:
  // Holds burst buffers that cannot be freed yet
  std::array<std::shared_ptr<AdvNetBurstParams>, 256> burst_bufs_;
  int     burst_buf_idx_ = 0;                // Index into burst_buf_idx_ of current burst
  int64_t ttl_bytes_recv_ = 0;               // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;                // Total packets received in operator
  int64_t aggr_pkts_recv_ = 0;               // Aggregate packets received in processing batch
  uint16_t nom_payload_size_;                // Nominal payload size (no headers)
  void **h_dev_ptrs_;                        // Host-pinned list of device pointers
  void *full_batch_data_h_;                  // Host-pinned aggregated batch
  void *full_batch_data_d_;                  // Device aggregated batch
  Parameter<bool> hds_;                      // Header-data split enabled
  Parameter<uint32_t> batch_size_;           // Batch size for one processing block
  Parameter<uint16_t> max_packet_size_;      // Maximum size of a single packet
};

}  // namespace holoscan::ops

class App : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    HOLOSCAN_LOG_INFO("Initializing advanced network operator");
    const auto [rx_en, tx_en] = holoscan::ops::adv_net_get_rx_tx_cfg_en(config());

    if (rx_en) {
      auto bench_rx     = make_operator<ops::AdvNetworkingBenchRxOp>("bench_rx",
                                                                      from_config("bench_rx"));
      auto adv_net_rx   = make_operator<ops::AdvNetworkOpRx>("adv_network_rx",
                                              from_config("advanced_network"),
                                              make_condition<BooleanCondition>("is_alive", true));
      add_flow(adv_net_rx, bench_rx, {{"burst_out", "burst_in"}});
    }
    if (tx_en) {
      auto bench_tx       = make_operator<ops::AdvNetworkingBenchTxOp>("bench_tx",
                                              from_config("bench_tx"),
                                              make_condition<BooleanCondition>("is_alive", true));
      auto adv_net_tx     = make_operator<ops::AdvNetworkOpTx>("adv_network_tx",
                                                              from_config("advanced_network"));
      add_flow(bench_tx, adv_net_tx, {{"burst_out", "burst_in"}});
    }
  }
};

int main(int argc, char** argv) {
  auto app = holoscan::make_application<App>();

  // Get the configuration
  if (argc < 2) {
    HOLOSCAN_LOG_ERROR("Usage: {} config_file", argv[0]);
    return -1;
  }

  auto config_path = std::filesystem::canonical(argv[0]).parent_path();
  config_path += "/" + std::string(argv[1]);
  app->config(config_path);

  app->run();

  return 0;
}
