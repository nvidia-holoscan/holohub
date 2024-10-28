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

class AdvNetworkingBenchDefaultRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvNetworkingBenchDefaultRxOp)

  AdvNetworkingBenchDefaultRxOp() = default;

  ~AdvNetworkingBenchDefaultRxOp() {
    HOLOSCAN_LOG_INFO(
        "Finished receiver with {}/{} bytes/packets received", ttl_bytes_recv_, ttl_pkts_recv_);

    HOLOSCAN_LOG_INFO("ANO benchmark RX op shutting down");
    adv_net_shutdown();
    adv_net_print_stats();
  }

  void initialize() override {
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultRxOp::initialize()");
    holoscan::Operator::initialize();

    // For this example assume all packets are the same size, specified in the config
    nom_payload_size_ = max_packet_size_.get() - header_size_.get();

    for (int n = 0; n < num_concurrent; n++) {
      cudaMalloc(&full_batch_data_d_[n], batch_size_.get() * nom_payload_size_);
      if (!gpu_direct_.get()) {
        cudaMallocHost(&full_batch_data_h_[n], batch_size_.get() * nom_payload_size_);
      } else {
        cudaMallocHost((void**)&h_dev_ptrs_[n], sizeof(void*) * batch_size_.get());
      }

      cudaStreamCreate(&streams_[n]);
      cudaEventCreate(&events_[n]);
    }

    if (hds_.get()) { assert(gpu_direct_.get()); }

    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultRxOp::initialize() complete");
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

  // Free buffers if CUDA processing/copy is complete
  void free_processed_packets() {
    // Iterate through the batches tracked for processing
    while (batch_q_.size() > 0) {
      const auto batch = batch_q_.front();
      // If CUDA processing/copy is complete, free the packets for all bursts in this batch
      if (cudaEventQuery(batch.evt) == cudaSuccess) {
        for (auto m = 0; m < batch.num_bursts; m++) {
          adv_net_free_all_pkts_and_burst(batch.bursts[m]);
        }
        batch_q_.pop();
      } else {
        // No need to check the next batch if the previous one is still being processed
        break;
      }
    }
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext& context) override {
    // If we processed a batch of packets in a previous compute call, that was done asynchronously,
    // and we'll need to free the packets eventually so the NIC can have space for the next bursts.
    // Ideally, we'd free the packets on a callback from CUDA, but that is slow. For that reason and
    // to keep it simple, we do that check right here on the next epoch of the operator.
    free_processed_packets();

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
      adv_net_free_all_pkts_and_burst(burst);
      return;
    }

    // Store burst structure
    cur_batch_.bursts[cur_batch_.num_bursts++] = burst;

    // Count packets received
    ttl_pkts_recv_ += adv_net_get_num_pkts(burst);

    // Track packet payloads for the current burst
    if (gpu_direct_.get()) {
      /* GPUDirect mode (needs to match if the ANO queue uses one or more memory regions)
       * Save off the GPU pointers into a host-pinned buffer (h_dev_ptrs_) to reassemble later.
       */
      if (hds_.get()) {
        // Header-Data-Split: header to CPU, payload to GPU
        // NOTE: current App assumes only two memory region segments, one for header (CPU),
        //       and one for payload (GPU).
        for (int p = 0; p < adv_net_get_num_pkts(burst); p++) {
          // Get pointers to payload data on GPU
          // NOTE: It's (1) here since the GPU memory region is second in the list for this queue.
          //       The first region (0) is for headers on CPU, ignored here.
          // NOTE: currently ordering pointers in the order packets come in. If headers had segment
          //       ID, the index in h_dev_ptrs_ should use that (instead of aggr_pkts_recv_ + p).
          h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_ + p] = adv_net_get_seg_pkt_ptr(burst, 1, p);

          // Count bytes received
          ttl_bytes_recv_ +=
              adv_net_get_seg_pkt_len(burst, 0, p) + adv_net_get_seg_pkt_len(burst, 1, p);

          // TODO: could free CPU packets (segment 0) now
        }
      } else {
        // Batched: headers and payload to GPU (queue memory regions should be a single GPU segment)
        for (int p = 0; p < adv_net_get_num_pkts(burst); p++) {
          // Get pointers to payload data on GPU (shifting by header size)
          // NOTE: currently ordering pointers in the order packets come in. If headers had segment
          //       ID, the index in h_dev_ptrs_ should use that (instead of aggr_pkts_recv_ + p).
          h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_ + p] =
              reinterpret_cast<uint8_t*>(adv_net_get_seg_pkt_ptr(burst, 0, p)) + header_size_.get();

          // Count bytes received
          ttl_bytes_recv_ += adv_net_get_seg_pkt_len(burst, 0, p);
        }
      }
    } else {
      /* CPU Mode (needs to match if the ANO queue uses no GPU memory regions)
       * Copy each packet payload in a continuous host-pinned buffer, copy of that larger buffer to
       * the GPU will occur later (copying each packet to GPU directly would be too expensive).
       *
       * NOTE: this assume huge pages memory regions. With host-pinned memory regions, this could be
       *       skipped, though probably not faster given the higher perf to write to huge pages.
       */

      // Calculate offset for the ANO burst
      // NOTE: we could keep track of each packet length and aggregate it instead.
      auto burst_offset = aggr_pkts_recv_ * nom_payload_size_;

      for (int p = 0; p < adv_net_get_num_pkts(burst); p++) {
        // Payload address (UDPIPV4Pkt: + 1 skips the header)
        auto payload_ptr = static_cast<UDPIPV4Pkt*>(adv_net_get_seg_pkt_ptr(burst, 0, p)) + 1;
        // Payload length (packet length minus header length)
        // NOTE: this should be equal to nom_payload_size_ as we assume the same length for all
        //       packets in this sample app
        auto pkt_len = adv_net_get_seg_pkt_len(burst, 0, p);
        auto payload_len = pkt_len - header_size_.get();

        // Copy payload to aggregated CPU buffers now
        memcpy((char*)full_batch_data_h_[cur_batch_idx_] + burst_offset + p * nom_payload_size_,
               payload_ptr,
               payload_len);

        // Count bytes received
        ttl_bytes_recv_ += pkt_len;

        // TODO: could free CPU packets now
      }
    }

     /* For each ANO batch (named burst), we might not want to right away send the packets to the
     * next operator, but maybe wait for more packets to come in, to make up what we call an
     * "App batch". While that increases the latency by needing more data to come in to continue,
     * it would allow collecting enough packets for reordering (not done here) to trigger the
     * downstream pipeline as soon as we have enough packets to do a full "message".
     * Increasing the burst size instead (ANO batch) would ensure the same, but allowing smaller
     * burst size will improve latency.
     *
     * There is also value in CPU mode or HDS mode: to reduce CPU memory usage by not holding onto
     * packets that can be freed earlier on (whether full packet buffers or headers only for HDS)
     *
     * Below, we check if we should wait to receive more packets from
     * the next burst before processing them in a batch.
     */
    aggr_pkts_recv_ += adv_net_get_num_pkts(burst);
    if (aggr_pkts_recv_ >= batch_size_.get()) {
      // Reset counter for the next app batch
      aggr_pkts_recv_ = 0;

      // Free buffers for packets which have already been aggregated to the GPU, in case
      // some of it got completed since the beginning of `compute`, for extra space in `batch_q_`.
      // NOTE: In CPU-only mode we can free earlier (after memcopy), but to keep it simple we free
      //       at the same point as we do in GPUDirect mode
      free_processed_packets();
      if (batch_q_.size() == num_concurrent) {
        // Not enough buffers available to process the packets, drop the packets from this burst
        HOLOSCAN_LOG_ERROR("Fell behind putting packet data in contiguous memory on GPU!");
        adv_net_free_all_pkts_and_burst(burst);
        return;
      }

      // Process/aggregate
      if (gpu_direct_.get()) {
        // GPUDirect mode: we copy the payload (referenced in h_dev_ptrs_)
        // to a contiguous memory buffer (full_batch_data_d_)
        // NOTE: there is no actual reordering since we use the same order as packets came in,
        //   but they would be reordered if h_dev_ptrs_ was filled based on packet sequence id.
        simple_packet_reorder(static_cast<uint8_t*>(full_batch_data_d_[cur_batch_idx_]),
                              h_dev_ptrs_[cur_batch_idx_],
                              nom_payload_size_,
                              batch_size_.get(),
                              streams_[cur_batch_idx_]);

      } else {
          // Non GPUDirect mode: we copy the payload on host-pinned memory (in full_batch_data_h_)
          // to a contiguous memory buffer on the GPU (full_batch_data_d_)
          // NOTE: there is no reordering support here at all
          cudaMemcpyAsync(full_batch_data_d_[cur_batch_idx_],
                          full_batch_data_h_[cur_batch_idx_],
                          batch_size_.get() * nom_payload_size_,
                          cudaMemcpyDefault,
                          streams_[cur_batch_idx_]);
      }

      /* Keep track of the CUDA work (reorder kernel or copy from CPU) so we do not process
       * more than `num_concurrent` batches at once, and to can free its associated packets
       * when the work is done.
       */
      cudaEventRecord(events_[cur_batch_idx_], streams_[cur_batch_idx_]);
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
  std::array<void*, num_concurrent> full_batch_data_h_;  // Host aggregated batch
  Parameter<bool> hds_;                                  // Header-data split enabled
  Parameter<bool> gpu_direct_;                           // GPUDirect enabled
  Parameter<uint32_t> batch_size_;                       // Batch size for one processing block
  Parameter<uint16_t> max_packet_size_;                  // Maximum size of a single packet
  Parameter<uint16_t> header_size_;                      // Header size of packet

  std::array<cudaStream_t, num_concurrent> streams_;
  std::array<cudaEvent_t, num_concurrent> events_;
};

}  // namespace holoscan::ops
