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
#include "holoscan/holoscan.hpp"
#include <queue>
#include <arpa/inet.h>
#include <assert.h>
#include <sys/time.h>

#define BURST_ACCESS_METHOD_RAW_PTR 0
#define BURST_ACCESS_METHOD_DIRECT_ACCESS 1

#define BURST_ACCESS_METHOD BURST_ACCESS_METHOD_RAW_PTR

using namespace holoscan::advanced_network;

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
                         static_cast<int>(_holoscan_cuda_err));                                 \
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

    HOLOSCAN_LOG_INFO("Advanced Networking Benchmark RX op shutting down");
    freeResources();
  }

  void initialize() override {
    cudaError_t cuda_error;
    HOLOSCAN_LOG_INFO("AdvNetworkingBenchDefaultRxOp::initialize()");
    holoscan::Operator::initialize();

    port_id_ = get_port_id(interface_name_.get());
    if (port_id_ == -1) {
      HOLOSCAN_LOG_ERROR("Invalid RX port {} specified in the config", interface_name_.get());
      exit(1);
    }

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
    spec.param<std::string>(interface_name_,
                            "interface_name",
                            "Port name",
                            "Name of the port to poll on from the advanced_network config",
                            "rx_port");
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
    spec.param<bool>(reorder_kernel_,
                     "reorder_kernel",
                     "Reorder kernel enabled",
                     "Enable reorder kernel",
                     true);
  }

  // Free buffers if CUDA processing/copy is complete
  void free_processed_packets() {
    // Iterate through the batches tracked for processing
    while (batch_q_.size() > 0) {
      const auto batch = batch_q_.front();
      // If CUDA processing/copy is complete, free the packets for all bursts in this batch
      if (cudaEventQuery(batch.evt) == cudaSuccess) {
        for (auto m = 0; m < batch.num_bursts; m++) {
          free_all_packets_and_burst_rx(batch.bursts[m]);
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

    BurstParams *burst;

    // In this example, we'll loop through all the rx queues of the interface
    // assuming we want to process the packets the same way for all queues
    const auto num_rx_queues = get_num_rx_queues(port_id_);
    for (int q = 0; q < num_rx_queues; q++) {
      auto status = get_rx_burst(&burst, port_id_, q);
      if (status != Status::SUCCESS) {
        HOLOSCAN_LOG_DEBUG("No RX burst available");
        continue;
      }

      auto burst_size = get_num_packets(burst);

      // Count packets received
      ttl_pkts_recv_ += burst_size;

      // Store burst structure
      cur_batch_.bursts[cur_batch_.num_bursts++] = burst;

      // Track packet payloads for the current burst
      if (gpu_direct_.get()) {
        /* GPUDirect mode (needs to match if the advanced_network queue uses 1 or more memory regions)
        * Save off the GPU pointers into a host-pinned buffer (h_dev_ptrs_) to reassemble later.
        */
        if (hds_.get()) {
          // Header-Data-Split: header to CPU, payload to GPU
          // NOTE: current App assumes only two memory region segments, one for header (CPU),
          //       and one for payload (GPU).
          for (int p = 0; p < burst_size; p++) {
            // Get pointers to payload data on GPU
            // NOTE: It's (1) here since the GPU memory region is second in the list for this queue.
            //       The first region (0) is for headers on CPU, ignored here.
            // NOTE: currently ordering pointers in the order packets come in. If headers had
            //       segment ID, the index in h_dev_ptrs_ should use that
            //       (instead of aggr_pkts_recv_ + p).
  #if (BURST_ACCESS_METHOD == BURST_ACCESS_METHOD_DIRECT_ACCESS)
            h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_ + p] = burst->pkts[1][p];
            ttl_bytes_recv_ += burst->pkt_lens[0][p] + burst->pkt_lens[1][p];
  #else
            h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_ + p] = get_segment_packet_ptr(burst, 1, p);
            ttl_bytes_recv_ +=
                get_segment_packet_length(burst, 0, p) + get_segment_packet_length(burst, 1, p);
  #endif
          }
        } else {
          // Batched: headers and payload to GPU (queue memory regions should be a single
          // GPU segment)
          for (int p = 0; p < burst_size; p++) {
            // Get pointers to payload data on GPU (shifting by header size)
            // NOTE: currently ordering pointers in the order packets come in. If headers had
            //       segment ID, the index in h_dev_ptrs_ should use that (instead of
            //       aggr_pkts_recv_ + p).
            h_dev_ptrs_[cur_batch_idx_][aggr_pkts_recv_ + p] =
                reinterpret_cast<uint8_t*>(get_segment_packet_ptr(burst, 0, p)) +
                header_size_.get();
            ttl_bytes_recv_ += get_segment_packet_length(burst, 0, p);
          }
        }
      } else {
        /* CPU Mode (needs to match if the advanced_network queue uses no GPU memory regions)
        * Copy each packet payload in a continuous host-pinned buffer, copy of that larger buffer to
        * the GPU will occur later (copying each packet to GPU directly would be too expensive).
        *
        * NOTE: this assume huge pages memory regions. With host-pinned memory regions, this could be
        *       skipped, though probably not faster given the higher perf to write to huge pages.
        */

        // Calculate offset for the burst
        // NOTE: we could keep track of each packet length and aggregate it instead.
        auto burst_offset = aggr_pkts_recv_ * nom_payload_size_;

        for (int p = 0; p < burst_size; p++) {
          // Payload address (UDPIPV4Pkt: + 1 skips the header)
  #if (BURST_ACCESS_METHOD == BURST_ACCESS_METHOD_DIRECT_ACCESS)
          auto payload_ptr = static_cast<UDPIPV4Pkt*>(burst->pkts[0][p]) + 1;
          // Payload length (packet length minus header length)
          // NOTE: this should be equal to nom_payload_size_ as we assume the same length for all
          //       packets in this sample app
          auto pkt_len = burst->pkt_lens[0][p];
  #else
          auto payload_ptr = static_cast<UDPIPV4Pkt*>(get_segment_packet_ptr(burst, 0, p)) + 1;
          // Payload length (packet length minus header length)
          // NOTE: this should be equal to nom_payload_size_ as we assume the same length for all
          //       packets in this sample app
          auto pkt_len = get_segment_packet_length(burst, 0, p);
  #endif
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

      /* For each incoming burst, we might not want to right away send the packets to the next
      * operator, but maybe wait for more packets to come in, to make up what we call a batch.
      * While that increases the latency by needing more data to come in to continue,
      * it would allow collecting enough packets for reordering (not done here) to trigger the
      * downstream pipeline as soon as we have enough packets to do a full "message".
      * Increasing the burst size instead would ensure the same, but allowing smaller
      * burst size will improve latency.
      *
      * There is also value in CPU mode or HDS mode: to reduce CPU memory usage by not holding onto
      * packets that can be freed earlier on (whether full packet buffers or headers only for HDS)
      *
      * Below, we check if we should wait to receive more packets from
      * the next burst before processing them in a batch.
      */
      aggr_pkts_recv_ += burst_size;
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
          for (auto m = 0; m < cur_batch_.num_bursts; m++) {
            ttl_packets_dropped_ += get_num_packets(cur_batch_.bursts[m]);
            free_all_packets_and_burst_rx(cur_batch_.bursts[m]);
          }
          cur_batch_.num_bursts = 0;
          CUDA_TRY(cudaDeviceSynchronize());
          return;
        }

        // Process/aggregate
        if (gpu_direct_.get()) {
          // GPUDirect mode: we copy the payload (referenced in h_dev_ptrs_)
          // to a contiguous memory buffer (full_batch_data_d_)
          // NOTE: there is no actual reordering since we use the same order as packets came in,
          //   but they would be reordered if h_dev_ptrs_ was filled based on packet sequence id.
          if (reorder_kernel_.get()) {
            simple_packet_reorder(static_cast<uint8_t*>(full_batch_data_d_[cur_batch_idx_]),
                                  h_dev_ptrs_[cur_batch_idx_],
                                  nom_payload_size_,
                                  batch_size_.get(),
                                  streams_[cur_batch_idx_]);
          }

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

        if (cudaGetLastError() != cudaSuccess) {
          HOLOSCAN_LOG_ERROR("CUDA error with {} packets in batch and {} bytes total",
                            batch_size_.get(),
                            batch_size_.get() * nom_payload_size_);
          exit(1);
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
  }

 private:
  // TODO: make configurable?
  static constexpr int num_concurrent = 10;  // Number of concurrent batches processing
  // TODO: could infer with (batch_size / burst size)
  static constexpr int MAX_BURSTS_PER_BATCH = 10;

  // Holds burst buffers that cannot be freed yet and CUDA event indicating when they can be freed
  struct BatchAggregationParams {
    std::array<BurstParams*, MAX_BURSTS_PER_BATCH> bursts;
    int num_bursts;
    cudaEvent_t evt;
  };

  int port_id_;                                    // Port ID to poll on
  BatchAggregationParams cur_batch_{};             // Parameters of current batch to process
  int cur_batch_idx_ = 0;                          // Current batch ID
  std::queue<BatchAggregationParams> batch_q_;     // Queue of batches being processed
  int64_t ttl_bytes_recv_ = 0;                     // Total bytes received in operator
  int64_t ttl_pkts_recv_ = 0;                      // Total packets received in operator
  int64_t aggr_pkts_recv_ = 0;                     // Aggregate packets received in processing batch
  int64_t ttl_packets_dropped_ = 0;                // Total packets dropped in operator
  uint16_t nom_payload_size_;                      // Nominal payload size (no headers)
  std::array<void**, num_concurrent> h_dev_ptrs_;  // Host-pinned list of device pointers
  std::array<void*, num_concurrent> full_batch_data_d_;  // Device aggregated batch
  std::array<void*, num_concurrent> full_batch_data_h_;  // Host aggregated batch
  Parameter<std::string> interface_name_;                // Port name from advanced_network config
  Parameter<bool> hds_;                                  // Header-data split enabled
  Parameter<bool> gpu_direct_;                           // GPUDirect enabled
  Parameter<uint32_t> batch_size_;                       // Batch size for one processing block
  Parameter<uint16_t> max_packet_size_;                  // Maximum size of a single packet
  Parameter<uint16_t> header_size_;                      // Header size of packet
  Parameter<bool> reorder_kernel_;                        // Reorder kernel enabled

  std::array<cudaStream_t, num_concurrent> streams_;
  std::array<cudaEvent_t, num_concurrent> events_;
};

}  // namespace holoscan::ops
