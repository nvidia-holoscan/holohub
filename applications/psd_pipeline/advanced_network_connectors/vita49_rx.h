/*
 * SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/std/complex>
#include "holoscan/holoscan.hpp"
#include "matx.h"
#include "adv_network_rx.h"

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

// Swapping logic borrowed from RedhawkSDR:
// https://github.com/RedhawkSDR/VITA49/blob/master/cpp/include/VRTMath.h#L77
inline uint16_t bswap_16_h(const uint16_t val) {
    return ((val & 0xff00) >> 8)
         | ((val & 0x00ff) << 8);
}

inline __device__ uint16_t bswap_16(const uint16_t val) {
    return ((val & 0xff00) >> 8)
         | ((val & 0x00ff) << 8);
}

inline int16_t bswap_16_h(const int16_t val) {
    const uint16_t v = bswap_16_h(*((uint16_t*)&val));
    return *((int16_t*)&v);
}

inline __device__ int16_t bswap_16(const int16_t val) {
    const uint16_t v = bswap_16(*((uint16_t*)&val));
    return *((int16_t*)&v);
}

inline uint32_t bswap_32_h(uint32_t val) {
    return ((val & 0xff000000) >> 24)
         | ((val & 0x00ff0000) >>  8)
         | ((val & 0x0000ff00) <<  8)
         | ((val & 0x000000ff) << 24);
}

inline __device__ uint32_t bswap_32(uint32_t val) {
    return ((val & 0xff000000) >> 24)
         | ((val & 0x00ff0000) >>  8)
         | ((val & 0x0000ff00) <<  8)
         | ((val & 0x000000ff) << 24);
}

inline int32_t bswap_32_h(int32_t val) {
    uint32_t v = bswap_32_h(*((uint32_t*)&val));
    return *((int32_t*)&v);
}

inline __device__ int32_t bswap_32(int32_t val) {
    uint32_t v = bswap_32(*((uint32_t*)&val));
    return *((int32_t*)&v);
}

inline float bswap_32_h(float val) {
    uint32_t v = bswap_32_h(*((uint32_t*)&val));
    return *((float*)&v);
}

inline __device__ float bswap_32(float val) {
    uint32_t v = bswap_32(*((uint32_t*)&val));
    return *((float*)&v);
}

inline uint64_t bswap_64_h(uint64_t val) {
    return ((val & __UINT64_C(0xff00000000000000)) >> 56)
         | ((val & __UINT64_C(0x00ff000000000000)) >> 40)
         | ((val & __UINT64_C(0x0000ff0000000000)) >> 24)
         | ((val & __UINT64_C(0x000000ff00000000)) >>  8)
         | ((val & __UINT64_C(0x00000000ff000000)) <<  8)
         | ((val & __UINT64_C(0x0000000000ff0000)) << 24)
         | ((val & __UINT64_C(0x000000000000ff00)) << 40)
         | ((val & __UINT64_C(0x00000000000000ff)) << 56);
}

inline __device__ uint64_t bswap_64(uint64_t val) {
    return ((val & __UINT64_C(0xff00000000000000)) >> 56)
         | ((val & __UINT64_C(0x00ff000000000000)) >> 40)
         | ((val & __UINT64_C(0x0000ff0000000000)) >> 24)
         | ((val & __UINT64_C(0x000000ff00000000)) >>  8)
         | ((val & __UINT64_C(0x00000000ff000000)) <<  8)
         | ((val & __UINT64_C(0x0000000000ff0000)) << 24)
         | ((val & __UINT64_C(0x000000000000ff00)) << 40)
         | ((val & __UINT64_C(0x00000000000000ff)) << 56);
}

inline int64_t bswap_64_h(int64_t val) {
    uint64_t v = bswap_64_h(*((uint64_t*)&val));
    return *((int64_t*)&v);
}

inline __device__ int64_t bswap_64(int64_t val) {
    uint64_t v = bswap_64(*((uint64_t*)&val));
    return *((int64_t*)&v);
}

inline double bswap_64_h(double val) {
    uint64_t v = bswap_64_h(*((uint64_t*)&val));
    return *((double*)&v);
}

inline __device__ double bswap_64(double val) {
    uint64_t v = bswap_64(*((uint64_t*)&val));
    return *((double*)&v);
}

inline uint32_t get_vrt_header_h(const VitaMetaData *meta) {
    return bswap_32_h(meta->vrt_header);
}

inline __device__ uint32_t get_vrt_header(const VitaMetaData *meta) {
    return bswap_32(meta->vrt_header);
}

inline uint32_t get_stream_id_h(const VitaMetaData *meta) {
    return bswap_32_h(meta->stream_id);
}

inline __device__ uint32_t get_stream_id(const VitaMetaData *meta) {
    return bswap_32(meta->stream_id);
}

inline uint32_t get_integer_time_h(const VitaMetaData *meta) {
    return bswap_32_h(meta->integer_time);
}

inline __device__ uint32_t get_integer_time(const VitaMetaData *meta) {
    return bswap_32(meta->integer_time);
}

inline uint64_t get_fractional_time_h(const VitaMetaData *meta) {
    return bswap_64_h(meta->fractional_time);
}

inline __device__ uint64_t get_fractional_time(const VitaMetaData *meta) {
    return bswap_64(meta->fractional_time);
}

inline uint32_t get_cif0_h(const ContextPacket *packet) {
    return bswap_32_h(packet->cif0);
}

inline double get_bandwidth_hz_h(const ContextPacket *packet) {
    return bswap_64_h(packet->bandwidth_hz) / (double)(__INT64_C(1) << BW_RADIX);
}

inline __device__ double get_bandwidth_hz(const ContextPacket *packet) {
    return bswap_64(packet->bandwidth_hz) / (double)(__INT64_C(1) << BW_RADIX);
}

inline double get_rf_ref_freq_hz_h(const ContextPacket *packet) {
    return bswap_64_h(packet->rf_ref_freq_hz) / (double)(__INT64_C(1) << FREQ_RADIX);
}

inline __device__ double get_rf_ref_freq_hz(const ContextPacket *packet) {
    return bswap_64(packet->rf_ref_freq_hz) / (double)(__INT64_C(1) << FREQ_RADIX);
}

inline float get_ref_level_dbm_h(const ContextPacket *packet) {
    return (int16_t)(bswap_32_h(packet->reference_level_dbm) & 0xFFFF) \
           / (float)(__INT16_C(1) << REF_LEVEL_RADIX);
}

inline __device__ float get_ref_level_dbm(const ContextPacket *packet) {
    return (int16_t)(bswap_32(packet->reference_level_dbm) & 0xFFFF) \
           / (float)(__INT16_C(1) << REF_LEVEL_RADIX);
}

inline float get_gain_1_db_h(const ContextPacket *packet) {
    return bswap_16_h(packet->gain_stage_1) / (float)(__INT16_C(1) << GAIN_RADIX);
}

inline __device__ float get_gain_1_db(const ContextPacket *packet) {
    return bswap_16(packet->gain_stage_1) / (float)(__INT16_C(1) << GAIN_RADIX);
}

inline float get_gain_2_db_h(const ContextPacket *packet) {
    return bswap_16_h(packet->gain_stage_2) / (float)(__INT16_C(1) << GAIN_RADIX);
}

inline __device__ float get_gain_2_db(const ContextPacket *packet) {
    return bswap_16(packet->gain_stage_2) / (float)(__INT16_C(1) << GAIN_RADIX);
}

inline double get_sample_rate_sps_h(const ContextPacket *packet) {
    return bswap_64_h(packet->sample_rate_sps) / (double)(__INT64_C(1) << SR_RADIX);
}

inline __device__ double get_sample_rate_sps(const ContextPacket *packet) {
    return bswap_64(packet->sample_rate_sps) / (double)(__INT64_C(1) << SR_RADIX);
}

inline bool context_changed_h(const ContextPacket *packet) {
    return bswap_32_h(packet->cif0) & 0x80000000;
}

inline __device__ bool context_changed(const ContextPacket *packet) {
    return bswap_32(packet->cif0) & 0x80000000;
}

namespace holoscan::ops {

class Vita49ConnectorOpRx : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(Vita49ConnectorOpRx)

  Vita49ConnectorOpRx() = default;

  ~Vita49ConnectorOpRx() {
    adv_net_shutdown();
    adv_net_print_stats();
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
  uint32_t num_packets_per_batch;

  // Holds burst buffers that cannot be freed yet
  struct RxMsg {
    std::array<std::shared_ptr<AdvNetBurstParams>, MAX_ANO_BATCHES> msg;
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

  std::vector<RxMsg> free_bufs(std::shared_ptr<struct Channel> channel);
  void free_bufs_and_emit_arrays(OutputContext& op_output, std::shared_ptr<struct Channel> channel);
  void process_channel_data(
          OutputContext& op_output,
          std::shared_ptr<AdvNetBurstParams> burst,
          uint16_t channel_num);
};  // Vita49ConnectorOpRx

}  // namespace holoscan::ops
