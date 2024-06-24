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
#include "adv_network_kernels.h"
#include "holoscan/holoscan.hpp"
#include <queue>
#include <arpa/inet.h>
#include <assert.h>
#include <sys/time.h>

namespace holoscan::ops {

class AdvNetworkingBenchDocaRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingBenchDocaRxOp)

  AdvNetworkingBenchDocaRxOp() = default;

  ~AdvNetworkingBenchDocaRxOp() {
    HOLOSCAN_LOG_INFO(
        "Finished receiver with {}/{} bytes/packets received", ttl_bytes_recv_, ttl_pkts_recv_);

    HOLOSCAN_LOG_INFO("ANO benchmark RX op shutting down");
    adv_net_shutdown();
    adv_net_print_stats();
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDocaRxOp::initialize()");
    holoscan::Operator::initialize();

    HOLOSCAN_LOG_INFO("holoscan::Operator::initialize() complete");

    // For this example assume all packets are the same size, specified in the config
    nom_payload_size_ = max_packet_size_.get() - header_size_.get();

    for (int n = 0; n < num_concurrent; n++) {
      cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * batch_size_.get());
      cudaStreamCreateWithFlags(&streams_[n], cudaStreamNonBlocking);
      cudaMallocAsync(&full_batch_data_d_[n], batch_size_.get() * nom_payload_size_, streams_[n]);
      cudaEventCreate(&events_[n]);
      cudaEventCreate(&events_start_[n]);
      // Warmup streams and kernel
      simple_packet_reorder(NULL, NULL, 1, 1, streams_[n]);
      cudaStreamSynchronize(streams_[n]);
    }

    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDocaRxOp::initialize() complete");
  }

  void setup(OperatorSpec& spec) override {
    spec.input<std::shared_ptr<AdvNetBurstParams>>("burst_in");
    // spec.param<bool>(hds_, "split_boundary", "Header-data split boundary",
    //     "Byte boundary where header and data is split", false);
    spec.param<uint32_t>(batch_size_,
                         "batch_size",
                         "Batch size",
                         "Batch size in packets for each processing epoch",
                         1000);
    spec.param<uint16_t>(max_packet_size_,
                         "max_packet_size",
                         "Max packet size",
                         "Maximum packet size expected from sender",
                         9100);
    spec.param<uint16_t>(header_size_,
                         "header_size",
                         "Header size",
                         "Header size on each packet from L4 and below",
                         42);
  }

  void free_bufs() {
    while (out_q.size() > 0) {
      const auto first = out_q.front();
      if (cudaEventQuery(first.evt) == cudaSuccess) {
        out_q.pop();
      } else {
        break;
      }
    }
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {
    int64_t ttl_bytes_in_cur_batch_ = 0;
    int pkt_idx = 0;
    bool complete = true;

    auto burst_opt = op_input.receive<std::shared_ptr<AdvNetBurstParams>>("burst_in");
    if (!burst_opt) {
      free_bufs();
      return;
    }

    auto burst = burst_opt.value();

    ttl_pkts_recv_ += adv_net_get_num_pkts(burst);

    // In config file, queue 0 is for all other non-UDP packets so we don't care
    if (adv_net_get_q_id(burst) == 0) {
      // HOLOSCAN_LOG_INFO("Ignoring packets on queue 0");
      return;
    }

    for (pkt_idx = 0; pkt_idx < adv_net_get_num_pkts(burst); pkt_idx++) {
      if (aggr_pkts_recv_ >= batch_size_.get()) {
        free_bufs();

        if (out_q.size() == num_concurrent) {
          HOLOSCAN_LOG_ERROR("Fell behind in processing on GPU!");
          return;
        }

        // HOLOSCAN_LOG_INFO("Launch order kernel, aggr_pkts_recv_ {} pkt_idx {} batch_size_.get()
        // {} cur_idx {}", aggr_pkts_recv_, pkt_idx, batch_size_.get(), cur_idx);
#if DEBUG_CUDA_TIMES == 1
        float et_ms = 0;
        cudaEventRecord(events_start_[cur_idx], streams_[cur_idx]);
#endif

        simple_packet_reorder(static_cast<uint8_t*>(full_batch_data_d_[cur_idx]),
                              h_dev_ptrs_[cur_idx],
                              nom_payload_size_,
                              batch_size_.get(),
                              streams_[cur_idx]);
#if DEBUG_CUDA_TIMES == 1
        cudaEventRecord(events_[cur_idx], streams_[cur_idx]);
        cudaEventSynchronize(events_[cur_idx]);
        cudaEventElapsedTime(&et_ms, events_start_[cur_idx], events_[cur_idx]);
        HOLOSCAN_LOG_INFO("aggr_pkts_recv_ {} et_ms {}", aggr_pkts_recv_, et_ms);
#endif
        cur_msg_.evt = events_[cur_idx];
        out_q.push(cur_msg_);
        cur_msg_.num_batches = 0;

        if (cudaGetLastError() != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("CUDA error with {} packets in batch and {} bytes total",
                             batch_size_.get(),
                             batch_size_.get() * nom_payload_size_);
          exit(1);
        }

        cur_idx = (++cur_idx % num_concurrent);
        aggr_pkts_recv_ = 0;
      }

      h_dev_ptrs_[cur_idx][aggr_pkts_recv_++] =
          reinterpret_cast<uint8_t*>(adv_net_get_pkt_ptr(burst, pkt_idx)) + header_size_.get();
    }

    ttl_bytes_in_cur_batch_ += adv_net_get_burst_tot_byte(burst);
    ttl_bytes_recv_ += ttl_bytes_in_cur_batch_;
  }

 private:
  static constexpr int num_concurrent = 4;    // Number of concurrent batches processing
  static constexpr int MAX_ANO_BATCHES = 10;  // Batches from ANO for one app batch

  // Holds burst buffers that cannot be freed yet
  struct RxMsg {
    std::array<std::shared_ptr<AdvNetBurstParams>, MAX_ANO_BATCHES> msg;
    int num_batches;
    cudaEvent_t evt;
  };

  RxMsg cur_msg_{};
  std::queue<RxMsg> out_q;
  int burst_buf_idx_ = 0;                          // Index into burst_buf_idx_ of current burst
  int64_t ttl_bytes_recv_ = 0;                     // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;                      // Total packets received in operator
  int64_t aggr_pkts_recv_ = 0;                     // Aggregate packets received in processing batch
  uint16_t nom_payload_size_;                      // Nominal payload size (no headers)
  std::array<void**, num_concurrent> h_dev_ptrs_;  // Host-pinned list of device pointers
  void* full_batch_data_h_;                        // Host-pinned aggregated batch
  std::array<void*, num_concurrent> full_batch_data_d_;  // Device aggregated batch
  Parameter<uint32_t> batch_size_;                       // Batch size for one processing block
  Parameter<uint16_t> max_packet_size_;                  // Maximum size of a single packet
  Parameter<uint16_t> header_size_;                      // Header size of packet
  static int s;
  std::array<cudaStream_t, num_concurrent> streams_;
  std::array<cudaEvent_t, num_concurrent> events_;
  std::array<cudaEvent_t, num_concurrent> events_start_;
  int cur_idx = 0;
};

}  // namespace holoscan::ops
