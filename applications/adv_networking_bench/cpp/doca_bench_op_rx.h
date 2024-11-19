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
#if ADV_NETWORK_MANAGER_WARMUP_KERNEL
      simple_packet_reorder(NULL, NULL, 1, 1, streams_[n]);
#endif
      cudaStreamSynchronize(streams_[n]);
    }

    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDocaRxOp::initialize() complete");
  }

  void setup(OperatorSpec& spec) override {
    spec.input<std::shared_ptr<AdvNetBurstParams>>("burst_in");
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

  // GPUNet IO has a ring buffer, older buffers get overridden without warning
  // as new packets come in so there is no need to free previous buffers. This method is still
  // needed to ensure we do not start processing too many batches in parallel.
  void free_batch_queue() {
    // Iterate through the batches tracked for copy/reordering
    while (batch_q_.size() > 0) {
      const auto batch = batch_q_.front();
      // If CUDA processing/copy is complete, stop tracking that batch
      // and leave space for the next batch in the queue
      if (cudaEventQuery(batch.evt) == cudaSuccess) {
        batch_q_.pop();
      } else {
        // No need to check the next batch if the previous one is still being processed
        break;
      }
    }
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {
    // If we processed a batch of packets in a previous compute call, that was done asynchronously,
    // and we'll need to know when that completes to have room to process more batches.
    // Ideally, we'd do that on a callback from CUDA, but that is slow. For that reason and
    // to keep it simple, we do that check right here on the next epoch of the operator.
    free_batch_queue();

    // Get new input burst (ANO batch of packets)
    auto burst_opt = op_input.receive<std::shared_ptr<AdvNetBurstParams>>("burst_in");
    if (!burst_opt) {
      HOLOSCAN_LOG_ERROR("No burst input");
      return;
    }
    auto burst = burst_opt.value();

    // If packets are coming in from our default (non-data) queue, free them and move on. This is
    // hardcoded to match the YAML config files in this sample app.
    // NOTE: we can't actually ignore all standard linux packets on a real network (with a switch),
    //       at least ARP packets should be processed, or delegate to linux for standard traffic.
    if (adv_net_get_q_id(burst) == 0) {
      HOLOSCAN_LOG_DEBUG("Ignoring packets on queue 0");
      return;
    }

    // Count packets received
    ttl_pkts_recv_ += adv_net_get_num_pkts(burst);

    // Iterate over packets on GPU
    for (int pkt_idx = 0; pkt_idx < adv_net_get_num_pkts(burst); pkt_idx++) {
      /* For each ANO batch (named burst), we might not want to right away send the packets to the
      * next operator, but maybe wait for more packets to come in, to make up what we call an
      * "App batch". While that increases the latency by needing more data to come in to continue,
      * it would allow collecting enough packets for reordering (not done here) to trigger the
      * downstream pipeline as soon as we have enough packets to do a full "message".
      * Increasing the burst size instead (ANO batch) would ensure the same, but allowing smaller
      * burst size will improve latency.
      *
      * Below, we check if we should wait to receive more packets from
      * the next burst before processing them in a batch.
      */
      if (aggr_pkts_recv_ >= batch_size_.get()) {
        // Reset counter for the next app batch
        aggr_pkts_recv_ = 0;

        // Free the queue for batches which have already been aggregated to the GPU again, in case
        // some of it got completed since the beginning of `compute`, for extra space in batch_q_.
        free_batch_queue();
        if (batch_q_.size() == num_concurrent) {
          HOLOSCAN_LOG_ERROR("Fell behind putting packet data in contiguous memory on GPU!");
          return;
        }

#if DEBUG_CUDA_TIMES == 1
        // Record kernel time
        cudaEventRecord(events_start_[cur_batch_idx_], streams_[cur_batch_idx_]);
#endif
        // CUDA kernel that copies the payload (referenced in h_dev_ptrs_)
        // to a contiguous memory buffer (full_batch_data_d_)
        // NOTE: there is no actual reordering since we use the same order as packets came in,
        //   but they would be reordered if h_dev_ptrs_ was filled based on packet sequence id.
        // HOLOSCAN_LOG_INFO("Launch order kernel, aggr_pkts_recv_ {} pkt_idx {} batch_size_.get()
        // {} cur_batch_idx_ {}", aggr_pkts_recv_, pkt_idx, batch_size_.get(), cur_batch_idx_);
        simple_packet_reorder(static_cast<uint8_t*>(full_batch_data_d_[cur_batch_idx_]),
                              h_dev_ptrs_[cur_batch_idx_],
                              nom_payload_size_,
                              batch_size_.get(),
                              streams_[cur_batch_idx_]);
#if DEBUG_CUDA_TIMES == 1
        cudaEventRecord(events_[cur_batch_idx_], streams_[cur_batch_idx_]);
        cudaEventSynchronize(events_[cur_batch_idx_]);
        float et_ms = 0;
        cudaEventElapsedTime(&et_ms, events_start_[cur_batch_idx_], events_[cur_batch_idx_]);
        HOLOSCAN_LOG_INFO("aggr_pkts_recv_ {} et_ms {}", aggr_pkts_recv_, et_ms);
#endif

        // Keep track of the CUDA work (reorder kernel) so we do not process
        // more than `num_concurrent` batches at once.
        cur_batch_.evt = events_[cur_batch_idx_];
        batch_q_.push(cur_batch_);

        // CUDA Error checking
        if (cudaGetLastError() != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("CUDA error with {} packets in batch and {} bytes total",
                             batch_size_.get(),
                             batch_size_.get() * nom_payload_size_);
          exit(1);
        }

        // Update structs for the next batch
        cur_batch_.num_bursts = 0;
        cur_batch_idx_ = (++cur_batch_idx_ % num_concurrent);

        // NOTE: output for the next operator would be full_batch_data_d_,
        // once the CUDA event is completed
      }

      // Get pointers to payload data on GPU (shifting by header size)
      // NOTE: currently ordering pointers in the order packets come in. If headers had segment
      //       ID, the index in h_dev_ptrs_ should use that (instead of aggr_pkts_recv_ + p).
      h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_++] =
          reinterpret_cast<uint8_t*>(adv_net_get_pkt_ptr(burst, pkt_idx)) + header_size_.get();
    }

    // Count bytes received
    ttl_bytes_recv_ += adv_net_get_burst_tot_byte(burst);
  }

 private:
  // TODO: make configurable?
  static constexpr int num_concurrent = 4;    // Number of concurrent batches processing
  // TODO: could infer with (batch_size / burst size)
  static constexpr int MAX_ANO_BURSTS = 10;   // Batches from ANO for one app batch

  // Holds burst buffers that cannot be freed yet and CUDA event indicating when they can be freed
  struct BatchAggregationParams {
    std::array<std::shared_ptr<AdvNetBurstParams>, MAX_ANO_BURSTS> bursts;
    int num_bursts;
    cudaEvent_t evt;
  };

  BatchAggregationParams cur_batch_{};             // Parameters of current batch to process
  int cur_batch_idx_ = 0;                          // Current batch ID
  std::queue<BatchAggregationParams> batch_q_;     // Queue of batches being processed
  int64_t ttl_bytes_recv_ = 0;                     // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;                      // Total packets received in operator
  int64_t aggr_pkts_recv_ = 0;                     // Aggregate packets received in processing batch
  uint16_t nom_payload_size_;                      // Nominal payload size (no headers)
  std::array<void**, num_concurrent> h_dev_ptrs_;  // Host-pinned list of device pointers
  std::array<void*, num_concurrent> full_batch_data_d_;  // Device aggregated batch
  Parameter<uint32_t> batch_size_;                       // Batch size for one processing block
  Parameter<uint16_t> max_packet_size_;                  // Maximum size of a single packet
  Parameter<uint16_t> header_size_;                      // Header size of packet

  std::array<cudaStream_t, num_concurrent> streams_;
  std::array<cudaEvent_t, num_concurrent> events_;
  std::array<cudaEvent_t, num_concurrent> events_start_;
};

}  // namespace holoscan::ops
