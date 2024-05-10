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
#pragma once

#include "common.h"
#include "adv_network_rx.h"

#include <linux/if_ether.h>
#include <linux/ip.h>
#include <linux/udp.h>
#include <arpa/inet.h>

// Compiler option that allows us to spoof packet metadata. This functionality
// can be useful when testing, where we have a packet generator that isn't
// transmitting data that isn't generating packets that use our data format.
#define SPOOF_PACKET_DATA      false
#define SPOOF_SAMPLES_PER_PKT  1024  // byte count must be less than 'max_packet_size' config

// Example IPV4 UDP packet using Linux headers
struct UDPIPV4Pkt {
  struct ethhdr eth;
  struct iphdr ip;
  struct udphdr udp;
  uint8_t payload[];
} __attribute__((packed));

// Tracks the status of filling an RF array
struct AdvBufferTracking {
  size_t pos;
  size_t pos_wrap;
  size_t buffer_size;
  int *sample_cnt_h;
  int *sample_cnt_d;
  bool *received_end_h;
  bool *received_end_d;

  AdvBufferTracking() = default;
  explicit AdvBufferTracking(const size_t _buffer_size)
    : pos(0), pos_wrap(0), buffer_size(_buffer_size) {
    // Reserve sample count
    cudaMallocHost((void **)&sample_cnt_h, buffer_size*sizeof(int));
    cudaMalloc((void **)&sample_cnt_d,     buffer_size*sizeof(int));
    memset(sample_cnt_h,     0, buffer_size*sizeof(int));
    cudaMemset(sample_cnt_d, 0, buffer_size*sizeof(int));

    // Reserve end-of-array signal
    cudaMallocHost((void **)&received_end_h, buffer_size*sizeof(bool));
    cudaMalloc((void **)&received_end_d,     buffer_size*sizeof(bool));
    memset(received_end_h,     0, buffer_size*sizeof(bool));
    cudaMemset(received_end_d, 0, buffer_size*sizeof(bool));
  }

  cudaError_t transferSamples(const cudaMemcpyKind kind, cudaStream_t stream) {
    void *src;
    void *dst;

    if (kind == cudaMemcpyHostToDevice) {
      src = sample_cnt_h;
      dst = sample_cnt_d;
    } else {
      src = sample_cnt_d;
      dst = sample_cnt_h;
    }
    return cudaMemcpyAsync(dst, src, buffer_size*sizeof(int), kind, stream);
  }

  cudaError_t transferEndArray(const cudaMemcpyKind kind, cudaStream_t stream) {
    void *src;
    void *dst;

    if (kind == cudaMemcpyHostToDevice) {
      src = received_end_h;
      dst = received_end_d;
    } else if (kind == cudaMemcpyDeviceToHost) {
      src = received_end_d;
      dst = received_end_h;
    } else {
      HOLOSCAN_LOG_ERROR("Unknown option {}", kind);
      return cudaErrorInvalidValue;
    }
    return cudaMemcpyAsync(dst, src, buffer_size*sizeof(bool), kind, stream);
  }

  // TODO: Faster way than two separate memcpy's?
  cudaError_t transfer(const cudaMemcpyKind kind, cudaStream_t stream) {
    cudaError_t err;
    err = transferSamples(kind, stream);
    if (err != cudaSuccess) {
      return err;
    }
    err = transferEndArray(kind, stream);
    if (err != cudaSuccess) {
      return err;
    }
    return cudaSuccess;
  }

  void increment() {
    received_end_h[pos_wrap] = false;
    sample_cnt_h[pos_wrap] = 0;
    pos++;
    pos_wrap = pos % buffer_size;
  }

  bool is_ready(const size_t samples_per_arr) {
    return received_end_h[pos_wrap] || sample_cnt_h[pos_wrap] >= samples_per_arr;
  }
};

namespace holoscan::ops {

class AdvConnectorOpRx : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(AdvConnectorOpRx)

  AdvConnectorOpRx() = default;

  ~AdvConnectorOpRx() {
    HOLOSCAN_LOG_INFO("Finished receiver with {}/{} bytes/packets received",
      ttl_bytes_recv_, ttl_pkts_recv_);
  }

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  static constexpr int num_concurrent  = 4;   // Number of concurrent batches processing
  static constexpr int MAX_ANO_BATCHES = 10;  // Batches from ANO for one app batch

  // Radar settings
  Parameter<uint16_t> buffer_size_;
  Parameter<uint16_t> num_channels_;
  Parameter<uint16_t> num_pulses_;
  Parameter<uint16_t> num_samples_;

  // Networking settings
  Parameter<bool> hds_;                  // Header-data split enabled
  Parameter<bool> gpu_direct_;           // GPUDirect enabled
  Parameter<uint32_t> batch_size_;       // Batch size for one processing block
  Parameter<uint16_t> max_packet_size_;  // Maximum size of a single packet
  Parameter<uint16_t> header_size_;      // Header size of packet

  // Holds burst buffers that cannot be freed yet
  struct RxMsg {
    std::array<std::shared_ptr<AdvNetBurstParams>, MAX_ANO_BATCHES> msg;
    int num_batches;
    cudaStream_t stream;
    cudaEvent_t evt;
  };
  std::vector<RxMsg> free_bufs();
  void free_bufs_and_emit_arrays(OutputContext& op_output);

  RxMsg cur_msg_{};
  std::queue<RxMsg> out_q;

  // Buffer memory and tracking
  std::array<void **, num_concurrent> h_dev_ptrs_;           // Host-pinned list of device pointers
  std::array<uint64_t **, num_concurrent> ttl_pkts_drop_;  // Total packets dropped by kernel

  // Concurrent batch structures
  std::array<cudaStream_t, num_concurrent> streams_;
  std::array<cudaEvent_t, num_concurrent> events_;
  int cur_idx = 0;

  // Holds burst buffers that cannot be freed yet
  int64_t ttl_bytes_recv_ = 0;           // Total bytes received in operator
  int64_t ttl_pkts_recv_  = 0;           // Total packets received in operator
  int64_t aggr_pkts_recv_ = 0;           // Aggregate packets received in processing batch
  uint16_t nom_payload_size_;            // Nominal payload size (no headers)

  size_t samples_per_arr;
  uint16_t pkts_per_pulse;
  uint16_t max_waveform_id;
  AdvBufferTracking buffer_track;
  tensor_t<complex_t, 4> rf_data;
  cudaStream_t proc_stream;
};  // AdvConnectorOpRx

}  // namespace holoscan::ops
