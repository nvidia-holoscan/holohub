/*
 * SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
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

#include "holoscan/holoscan.hpp"
#include "matx.h"
#include "advanced_network/common.h"

using namespace holoscan::advanced_network;
using namespace matx;
using complex = cuda::std::complex<float>;

struct VitaMetaData {
    uint32_t vrt_header;
    uint32_t stream_id;
    uint32_t integer_time;
    uint64_t fractional_time;
} __attribute__((__packed__));

struct ContextPacket {
    struct VitaMetaData metadata;
    uint32_t cif0;
    int64_t bandwidth_hz;
    int64_t rf_ref_freq_hz;
    int32_t reference_level_dbm;
    int16_t gain_stage_2;
    int16_t gain_stage_1;
    int64_t sample_rate_sps;
    uint64_t sig_data_payload_fmt;
} __attribute__((__packed__));

struct ContextData {
    bool context_changed;
    double bandwidth_hz;
    double rf_ref_freq_hz;
    float reference_level_dbm;
    float gain_stage_1_db;
    float gain_stage_2_db;
    double sample_rate_sps;
};

#define BW_RADIX 20
#define FREQ_RADIX 20
#define SR_RADIX 20
#define REF_LEVEL_RADIX 7
#define GAIN_RADIX 7

namespace holoscan::ops {

class Vita49ConnectorOpRx : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(Vita49ConnectorOpRx)

  Vita49ConnectorOpRx() = default;

  ~Vita49ConnectorOpRx() {
    shutdown();
    print_stats();
  }

  void setup(OperatorSpec& spec) override;
  void initialize() override;
  void compute(InputContext& op_input,
               OutputContext& op_output,
               ExecutionContext& context) override;
  void stop() override;

 private:
  static constexpr int num_concurrent  = 4;   // Number of concurrent batches processing
  static constexpr int MAX_ANO_BATCHES = 20;  // Batches from ANO for one app batch

  Parameter<uint16_t> num_complex_samples_per_packet_;
  Parameter<uint16_t> num_packets_per_fft_;
  Parameter<uint16_t> num_ffts_per_batch_;
  Parameter<uint16_t> num_simul_batches_;
  Parameter<uint16_t> num_channels_;
  Parameter<std::string> interface_name_;
  int port_id_;
  uint32_t num_packets_per_batch;

  // Holds burst buffers that cannot be freed yet
  struct RxMsg {
    std::array<BurstParams *, MAX_ANO_BATCHES> msg;
    int num_batches;
    cudaStream_t stream;
    cudaEvent_t evt;
  };

  struct Channel {
    uint16_t channel_num;
    int cur_idx = 0;
    tensor_t<complex, 3> rf_data;
    std::array<void **, num_concurrent> h_dev_ptrs;
    std::array<cudaStream_t, num_concurrent> streams;
    std::array<cudaEvent_t, num_concurrent> events;
    struct ContextData current_context;
    struct VitaMetaData current_meta;
    bool context_received = false;
    bool meta_set = false;
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
          BurstParams *burst,
          uint16_t channel_num);
};  // Vita49ConnectorOpRx

}  // namespace holoscan::ops
