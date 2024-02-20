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

#include <cuda/std/complex>
#include "holoscan/holoscan.hpp"
#include "matx.h"

using namespace matx;
using float_t = float;
using complex_t = cuda::std::complex<float_t>;

// Meta data for received signal
struct RfMetaData {
  uint16_t sample_idx;
  uint16_t waveform_id;
  uint16_t channel_idx;
  uint16_t pulse_idx;
  uint16_t pkt_samples;
  uint16_t end_array;
} __attribute__((__packed__));

// Represents a single channel of an RF transmission
struct RFChannel {
  tensor_t<complex_t, 2> data;
  uint16_t waveform_id;
  uint16_t channel_id;
  cudaStream_t stream;

  RFChannel(tensor_t<complex_t, 2> _data,
            uint16_t _waveform_id,
            uint16_t _channel_id,
            cudaStream_t _stream)
    : data{_data}, waveform_id{_waveform_id}, channel_id{_channel_id}, stream{_stream} {}
};

// Represents a single RF transmission
struct RFArray {
  tensor_t<complex_t, 3> data;
  uint16_t waveform_id;
  cudaStream_t stream;

  RFArray(tensor_t<complex_t, 3> _data, uint16_t _waveform_id, cudaStream_t _stream)
    : data{_data}, waveform_id{_waveform_id}, stream{_stream} {}
};

// Used to represent the data passed over the network
struct RFPacket {
 private:
  uint8_t *payload;

 public:
  static const size_t sample_offset     = 0;
  static const size_t waveformid_offset = sizeof(uint16_t) + sample_offset;
  static const size_t channel_offset    = sizeof(uint16_t) + waveformid_offset;
  static const size_t pulse_offset      = sizeof(uint16_t) + channel_offset;
  static const size_t num_sample_offset = sizeof(uint16_t) + pulse_offset;
  static const size_t end_array_offset  = sizeof(uint16_t) + num_sample_offset;
  // Setting the payload offset this way will prevent misaligned memory upon receipt
  static const size_t payload_offset    = sizeof(complex_t) * (
    (sizeof(uint16_t) + end_array_offset) / sizeof(complex_t) + 1);

 public:
  static size_t header_size() {
    return payload_offset;
  }
  static size_t payload_size(const size_t num_samples) {
    return num_samples * sizeof(complex_t);
  }
  static size_t packet_size(const size_t num_samples) {
    return header_size() + payload_size(num_samples);
  }

  RFPacket() = default;
  explicit RFPacket(uint8_t *buf_ptr) : payload{buf_ptr} {}

  // Set accessors
  void set_sample_idx(uint16_t sample_idx) {
    memcpy(&payload[sample_offset], &sample_idx, sizeof(uint16_t));
  }
  void set_waveform_id(uint16_t waveform_id) {
    memcpy(&payload[waveformid_offset], &waveform_id, sizeof(uint16_t));
  }
  void set_channel_idx(uint16_t channel_idx) {
    memcpy(&payload[channel_offset], &channel_idx, sizeof(uint16_t));
  }
  void set_pulse_idx(uint16_t pulse_idx) {
    memcpy(&payload[pulse_offset], &pulse_idx, sizeof(uint16_t));
  }
  void set_num_samples(uint16_t num_samples) {
    memcpy(&payload[num_sample_offset], &num_samples, sizeof(uint16_t));
  }
  void set_end_array(uint16_t is_end) {
    memcpy(&payload[end_array_offset], &is_end, sizeof(uint16_t));
  }
  void set_payload(complex_t *rf_data, cudaStream_t stream) {
    cudaMemcpyAsync(&payload[payload_offset],
                    rf_data,
                    payload_size(get_num_samples()),
                    cudaMemcpyDeviceToHost,
                    stream);
  }

  // Get accessors
  uint16_t get_sample_idx() {
    return *reinterpret_cast<uint16_t *>(payload + sample_offset);
  }
  uint16_t get_waveform_id() {
    return *reinterpret_cast<uint16_t *>(payload + waveformid_offset);
  }
  uint16_t get_channel_idx() {
    return *reinterpret_cast<uint16_t *>(payload + channel_offset);
  }
  uint16_t get_pulse_idx() {
    return *reinterpret_cast<uint16_t *>(payload + pulse_offset);
  }
  uint16_t get_num_samples() {
    return *reinterpret_cast<uint16_t *>(payload + num_sample_offset);
  }
  uint16_t get_end_array() {
    return *reinterpret_cast<uint16_t *>(payload + end_array_offset);
  }
  uint8_t* get_ptr() {
    return payload;
  }
  uint8_t* data() {
    return &payload[payload_offset];
  }

  void get_payload(uint8_t *ptr, const bool host) {
    if (host) {
      memcpy(ptr, payload, packet_size(get_num_samples()));
    } else {
      cudaMemcpyAsync(ptr,
                      payload,
                      packet_size(get_num_samples()),
                      cudaMemcpyHostToDevice);
    }
  }
};

// Compute the number of packets expected for an RF transmission
inline size_t packets_per_pulse(const size_t packet_size,
                                const size_t num_samples) {
  const size_t samples_per_pkt = (packet_size - RFPacket::header_size()) / sizeof(complex_t);
  return std::ceil(static_cast<double>(num_samples) / samples_per_pkt);
}

// Compute the number of packets expected for an RF transmission
inline size_t packets_per_channel(const size_t packet_size,
                                  const size_t num_pulses,
                                  const size_t num_samples) {
  return num_pulses * packets_per_pulse(packet_size, num_samples);
}

// Compute the number of packets expected for an RF transmission
inline size_t packets_per_array(const size_t packet_size,
                                const size_t num_pulses,
                                const size_t num_channels,
                                const size_t num_samples) {
  return num_channels * packets_per_channel(packet_size, num_pulses, num_samples);
}
