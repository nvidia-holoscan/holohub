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

#define BURST_ACCESS_METHOD_SHARED_PTR 0
#define BURST_ACCESS_METHOD_RAW_PTR 1
#define BURST_ACCESS_METHOD_DIRECT_ACCESS 2

#define BURST_ACCESS_METHOD BURST_ACCESS_METHOD_DIRECT_ACCESS

namespace holoscan::ops {

#define CUDA_TRY(stmt)                                                                          \
  ({                                                                                            \
    cudaError_t _holoscan_cuda_err = stmt;                                                      \
    if (cudaSuccess != _holoscan_cuda_err) {                                                    \
      HOLOSCAN_LOG_ERROR("CUDA Runtime call %s in line %d of file %s failed with '%s' (%d).\n", \
                         #stmt,                                                                 \
                         __LINE__,                                                              \
                         __FILE__,                                                              \
                         cudaGetErrorString(_holoscan_cuda_err),                                \
                         _holoscan_cuda_err);                                                   \
    }                                                                                           \
    _holoscan_cuda_err;                                                                         \
  })

#define ADV_NETWORK_MANAGER_WARMUP_KERNEL 1

class AdvNetworkingBenchDefaultRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingBenchDefaultRxOp)

  AdvNetworkingBenchDefaultRxOp() = default;

  ~AdvNetworkingBenchDefaultRxOp() {
    HOLOSCAN_LOG_INFO("Finished receiver with {}/{} bytes/packets received and {} packets dropped",
                      ttl_bytes_recv_,
                      ttl_pkts_recv_,
                      ttl_packets_dropped_);

    HOLOSCAN_LOG_INFO("ANO benchmark RX op shutting down");
    freeResources();
  }

  void initialize() override {
    cudaError_t cuda_error;
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultRxOp::initialize()");
    holoscan::Operator::initialize();

    // For this example assume all packets are the same size, specified in the config
    nom_payload_size_ = max_packet_size_.get() - header_size_.get();

    for (int n = 0; n < num_concurrent; n++) {
      cuda_error =
          CUDA_TRY(cudaMalloc(&full_batch_data_d_[n], batch_size_.get() * nom_payload_size_));
      if (cudaSuccess != cuda_error) {
        throw std::runtime_error("Could not allocate cuda memory for full_batch_data_d_[n]");
      }

      if (!gpu_direct_.get()) {
        cuda_error =
            CUDA_TRY(cudaMallocHost(&full_batch_data_h_[n], batch_size_.get() * nom_payload_size_));
        if (cudaSuccess != cuda_error) {
          throw std::runtime_error("Could not allocate cuda memory for full_batch_data_h_[n]");
        }
      } else {
        cuda_error =
            CUDA_TRY(cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * batch_size_.get()));
        if (cudaSuccess != cuda_error) {
          throw std::runtime_error("Could not allocate cuda memory for h_dev_ptrs_");
        }
      }
      cudaStreamCreate(&streams_[n]);
      cudaEventCreate(&events_[n]);
      // Warmup streams and kernel
#if ADV_NETWORK_MANAGER_WARMUP_KERNEL
      simple_packet_reorder(NULL, NULL, 1, 1, streams_[n]);
      cudaStreamSynchronize(streams_[n]);
#endif
    }
    if (hds_.get()) { assert(gpu_direct_.get()); }

    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultRxOp::initialize() complete");
  }

  void freeResources() {
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultRxOp::freeResources() start");
    for (int n = 0; n < num_concurrent; n++) {
      if (full_batch_data_d_[n]) { cudaFree(full_batch_data_d_[n]); }
      if (full_batch_data_h_[n]) { cudaFreeHost(full_batch_data_h_[n]); }
      if (h_dev_ptrs_[n]) { cudaFreeHost(h_dev_ptrs_[n]); }
      if (streams_[n]) { cudaStreamDestroy(streams_[n]); }
      if (events_[n]) { cudaEventDestroy(events_[n]); }
    }
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultRxOp::freeResources() complete");
  }

  void setup(OperatorSpec& spec) override {
    spec.input<std::shared_ptr<AdvNetBurstParams>>("burst_in");
    spec.param<bool>(hds_,
                     "split_boundary",
                     "Header-data split boundary",
                     "Byte boundary where header and data is split",
                     false);
    spec.param<bool>(gpu_direct_,
                     "gpu_direct",
                     "GPUDirect enabled",
                     "Byte boundary where header and data is split",
                     false);
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
        for (auto m = 0; m < first.num_batches; m++) {
          adv_net_free_all_pkts_and_burst(first.msg[m]);
        }
        out_q.pop();
      } else {
        break;
      }
    }
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {
    int64_t ttl_bytes_in_cur_batch_ = 0;

    auto burst_opt = op_input.receive<std::shared_ptr<AdvNetBurstParams>>("burst_in");
    if (!burst_opt) {
      free_bufs();
      return;
    }

#if (BURST_ACCESS_METHOD == BURST_ACCESS_METHOD_SHARED_PTR)
    auto burst = burst_opt.value();
#else
    auto burst_shared = burst_opt.value();
    auto burst = burst_shared.get();
#endif

    auto burst_size = adv_net_get_num_pkts(burst);
    ttl_pkts_recv_ += burst_size;

    // If packets are coming in from our non-GPUDirect queue, free them and move on
    if (adv_net_get_q_id(burst) == 0) {
      adv_net_free_all_pkts_and_burst(burst);
      return;
    }

    /* Header data split saves off the GPU pointers into a host-pinned buffer to reassemble later.
     * Once enough packets are aggregated, a reorder kernel is launched. In CPU-only mode the
     * entire burst buffer pointer is saved and freed once an entire batch is received.
     */
    if (gpu_direct_.get()) {
      int64_t bytes_in_batch = 0;
      if (hds_.get()) {
        for (int p = 0; p < burst_size; p++) {
#if (BURST_ACCESS_METHOD == BURST_ACCESS_METHOD_DIRECT_ACCESS)
          h_dev_ptrs_[cur_idx][aggr_pkts_recv_ + p] = burst->pkts[1][p];
          ttl_bytes_in_cur_batch_ += burst->pkt_lens[0][p] + burst->pkt_lens[1][p];
#else
          h_dev_ptrs_[cur_idx][aggr_pkts_recv_ + p] = adv_net_get_seg_pkt_ptr(burst, 1, p);
          ttl_bytes_in_cur_batch_ +=
              adv_net_get_seg_pkt_len(burst, 0, p) + adv_net_get_seg_pkt_len(burst, 1, p);
#endif
        }
      } else {
        for (int p = 0; p < burst_size; p++) {
#if (BURST_ACCESS_METHOD == BURST_ACCESS_METHOD_DIRECT_ACCESS)
          h_dev_ptrs_[cur_idx][aggr_pkts_recv_ + p] =
              reinterpret_cast<uint8_t*>(burst->pkts[0][p]) + header_size_.get();
          ttl_bytes_in_cur_batch_ += burst->pkt_lens[0][p];
#else
          h_dev_ptrs_[cur_idx][aggr_pkts_recv_ + p] =
              reinterpret_cast<uint8_t*>(adv_net_get_seg_pkt_ptr(burst, 0, p)) + header_size_.get();
          ttl_bytes_in_cur_batch_ += adv_net_get_seg_pkt_len(burst, 0, p);
#endif
        }
      }

      ttl_bytes_recv_ += ttl_bytes_in_cur_batch_;
    } else {
      auto batch_offset = aggr_pkts_recv_ * nom_payload_size_;
      for (int p = 0; p < burst_size; p++) {
#if (BURST_ACCESS_METHOD == BURST_ACCESS_METHOD_DIRECT_ACCESS)
        auto pkt = static_cast<UDPIPV4Pkt*>(burst->pkts[0][p]);
        auto len = burst->pkt_lens[0][p] - header_size_.get();
#else
        auto pkt = static_cast<UDPIPV4Pkt*>(adv_net_get_seg_pkt_ptr(burst, 0, p));
        auto len = adv_net_get_seg_pkt_len(burst, 0, p) - header_size_.get();
#endif
        memcpy((char*)full_batch_data_h_[cur_idx] + batch_offset + p * nom_payload_size_,
               pkt + 1,
               len);

        ttl_bytes_recv_ += len + sizeof(UDPIPV4Pkt);
        ttl_bytes_in_cur_batch_ += len + sizeof(UDPIPV4Pkt);
      }
    }

    aggr_pkts_recv_ += burst_size;
#if (BURST_ACCESS_METHOD == BURST_ACCESS_METHOD_SHARED_PTR)
    cur_msg_.msg[cur_msg_.num_batches++] = burst;
#else
    cur_msg_.msg[cur_msg_.num_batches++] = burst_shared;
#endif

    if (aggr_pkts_recv_ >= batch_size_.get()) {
      // Do some work on full_batch_data_h_ or full_batch_data_d_
      aggr_pkts_recv_ = 0;

      // In CPU-only mode we can free earlier, but to keep it simple we free at the same point
      // as we do in GPU-only mode
      free_bufs();

      if (out_q.size() == num_concurrent) {
        HOLOSCAN_LOG_ERROR("Fell behind in processing on GPU!");
        for (auto m = 0; m < cur_msg_.num_batches; m++) {
          ttl_packets_dropped_ += adv_net_get_num_pkts(cur_msg_.msg[m]);
          adv_net_free_all_pkts_and_burst(cur_msg_.msg[m]);
        }
        cur_msg_.num_batches = 0;
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
          std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }
        return;
      }

      if (gpu_direct_.get()) {
        simple_packet_reorder(static_cast<uint8_t*>(full_batch_data_d_[cur_idx]),
                              h_dev_ptrs_[cur_idx],
                              nom_payload_size_,
                              batch_size_.get(),
                              streams_[cur_idx]);

        if (cudaGetLastError() != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("CUDA error with {} packets in batch and {} bytes total",
                             batch_size_.get(),
                             batch_size_.get() * nom_payload_size_);
          exit(1);
        }
      } else {
        if (out_q.size() == num_concurrent) {
          HOLOSCAN_LOG_ERROR("Fell behind in copying to the GPU!");
          for (auto m = 0; m < cur_msg_.num_batches; m++) {
            ttl_packets_dropped_ += adv_net_get_num_pkts(cur_msg_.msg[m]);
            adv_net_free_all_pkts_and_burst(cur_msg_.msg[m]);
          }
          cur_msg_.num_batches = 0;
          cudaError_t err = cudaDeviceSynchronize();
          if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
          }
          return;
        }
        cudaMemcpyAsync(full_batch_data_d_[cur_idx],
                        full_batch_data_h_[cur_idx],
                        batch_size_.get() * nom_payload_size_,
                        cudaMemcpyDefault,
                        streams_[cur_idx]);
      }

      cudaEventRecord(events_[cur_idx], streams_[cur_idx]);

      cur_msg_.evt = events_[cur_idx];
      out_q.push(cur_msg_);
      cur_msg_.num_batches = 0;

      cur_idx = (++cur_idx % num_concurrent);
    }
  }

 private:
  static constexpr int num_concurrent = 10;   // Number of concurrent batches processing
  static constexpr int MAX_ANO_BATCHES = 10;  // Batches from ANO for one app batch

  // Holds burst buffers that cannot be freed yet
  struct RxMsg {
    std::array<std::shared_ptr<AdvNetBurstParams>, MAX_ANO_BATCHES> msg;
    int num_batches;
    void* full_batch_data_h_;
    cudaEvent_t evt;
  };

  RxMsg cur_msg_{};
  std::queue<RxMsg> out_q;
  int burst_buf_idx_ = 0;                          // Index into burst_buf_idx_ of current burst
  int64_t ttl_bytes_recv_ = 0;                     // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;                      // Total packets received in operator
  int64_t aggr_pkts_recv_ = 0;                     // Aggregate packets received in processing batch
  int64_t ttl_packets_dropped_ = 0;                // Total packets dropped in operator
  uint16_t nom_payload_size_;                      // Nominal payload size (no headers)
  std::array<void**, num_concurrent> h_dev_ptrs_;  // Host-pinned list of device pointers
  std::array<void*, num_concurrent> full_batch_data_d_;  // Device aggregated batch
  std::array<void*, num_concurrent> full_batch_data_h_;  // Host aggregated batch
  Parameter<bool> hds_;                                  // Header-data split enabled
  Parameter<bool> gpu_direct_;                           // GPUDirect enabled
  Parameter<uint32_t> batch_size_;                       // Batch size for one processing block
  Parameter<uint16_t> max_packet_size_;                  // Maximum size of a single packet
  Parameter<uint16_t> header_size_;                      // Header size of packet

  std::array<cudaStream_t, num_concurrent> streams_;
  std::array<cudaEvent_t, num_concurrent> events_;
  int cur_idx = 0;
};

}  // namespace holoscan::ops