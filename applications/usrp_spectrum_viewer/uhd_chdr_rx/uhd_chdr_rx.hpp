// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <queue>
#include <utility>
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
  Parameter<uint16_t> num_complex_samples_per_packet_;
  Parameter<uint16_t> num_packets_per_output_;
  Parameter<uint16_t> num_outputs_per_batch_;
  Parameter<uint16_t> num_buffered_batches_;
  Parameter<uint16_t> num_channels_;
  Parameter<std::string> interface_name_;
  Parameter<bool> log_data_;
  Parameter<bool> log_packets_;
  int port_id_ = -1;
  uint64_t num_packets_per_batch_ = 0;
  uint16_t emit_start_ = 0;

  // Owns both the converted tensor and the DAQIRI bursts backing an in-flight
  // conversion. The bursts are released once evt reports completion; the
  // tensor remains alive in the emitted message until all consumers finish.
  struct RxMsg {
    RxMsg(std::vector<daqiri::BurstParams*>&& message_bursts,
          matx::tensor_t<complex, 2>&& message_data,
          cudaStream_t message_stream,
          cudaEvent_t message_event)
        : bursts(std::move(message_bursts)),
          data(std::move(message_data)),
          stream(message_stream),
          evt(message_event) {}

    RxMsg(const RxMsg&) = delete;
    RxMsg& operator=(const RxMsg&) = delete;
    RxMsg(RxMsg&&) = default;
    RxMsg& operator=(RxMsg&&) = delete;

    std::vector<daqiri::BurstParams*> bursts;
    matx::tensor_t<complex, 2> data;
    cudaStream_t stream;
    cudaEvent_t evt;
  };

  struct Channel {
    uint16_t channel_num;
    size_t cur_idx = 0;
    std::vector<void**> h_dev_ptrs;
    cudaStream_t stream = nullptr;
    std::vector<cudaEvent_t> events;
    std::vector<daqiri::BurstParams*> current_bursts;
    std::queue<RxMsg> out_q;
    std::optional<uint16_t> next_sequence;
    uint64_t ttl_bytes_recv = 0;
    uint64_t ttl_pkts_recv = 0;
    uint64_t aggr_pkts_recv = 0;
  };

  std::vector<std::shared_ptr<struct Channel>> channel_list;

  std::optional<RxMsg> free_buf(std::shared_ptr<struct Channel> channel);
  bool free_bufs_and_emit_arrays(OutputContext& op_output, std::shared_ptr<struct Channel> channel);
  void process_channel_data(daqiri::BurstParams* burst, uint16_t channel_num);
  void discard_current_batch(const std::shared_ptr<struct Channel>& channel);
};  // UhdChdrRxOp

}  // namespace holoscan::ops
