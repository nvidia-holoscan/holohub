// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <queue>
#include <vector>

#include <daqiri/daqiri.h>

#include "holoscan/holoscan.hpp"
#include "matx.h"

using complex = cuda::std::complex<float>;

namespace holoscan::ops {

class UhdChdrRxOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(UhdChdrRxOp)

  UhdChdrRxOp() = default;

  ~UhdChdrRxOp() = default;

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  static constexpr int num_concurrent  = 4;   // Number of concurrent batches processing
  static constexpr int MAX_DAQIRI_BATCHES = 20;  // DAQIRI bursts for one app batch

  Parameter<uint16_t> num_complex_samples_per_packet_;
  Parameter<uint16_t> num_packets_per_output_;
  Parameter<uint16_t> num_outputs_per_batch_;
  Parameter<uint16_t> num_buffered_batches_;
  Parameter<uint16_t> num_channels_;
  Parameter<std::string> interface_name_;
  Parameter<bool> log_data_;
  Parameter<bool> log_packets_;
  int port_id_;
  uint32_t num_packets_per_batch;
  uint16_t emit_start_ = 0;

  // Holds burst buffers that cannot be freed yet
  struct RxMsg {
    std::array<daqiri::BurstParams *, MAX_DAQIRI_BATCHES> msg;
    int num_batches;
    int buf_idx;
    cudaStream_t stream;
    cudaEvent_t evt;
  };

  struct Channel {
    uint16_t channel_num;
    int cur_idx = 0;
    matx::tensor_t<complex, 3> rf_data;
    std::array<void **, num_concurrent> h_dev_ptrs;
    std::array<cudaStream_t, num_concurrent> streams;
    std::array<cudaEvent_t, num_concurrent> events;
    RxMsg cur_msg{};
    std::queue<RxMsg> out_q;
    uint64_t ttl_bytes_recv = 0;
    uint64_t ttl_pkts_recv = 0;
    uint64_t aggr_pkts_recv = 0;
  };

  std::vector<std::shared_ptr<struct Channel>> channel_list;

  std::optional<RxMsg> free_buf(std::shared_ptr<struct Channel> channel);
  bool free_bufs_and_emit_arrays(OutputContext& op_output, std::shared_ptr<struct Channel> channel);
  void process_channel_data(
          OutputContext& op_output,
          daqiri::BurstParams *burst,
          uint16_t channel_num);
};  // UhdChdrRxOp

}  // namespace holoscan::ops
