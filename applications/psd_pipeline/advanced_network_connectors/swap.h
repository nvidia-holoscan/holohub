/*
 * SPDX-FileCopyrightText: 2025 Valley Tech Systems, Inc.
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

// Swapping logic borrowed from RedhawkSDR:
// https://github.com/RedhawkSDR/VITA49/blob/master/cpp/include/VRTMath.h#L77
inline uint16_t bswap_16_h(const uint16_t val) {
    return ((val & 0xff00) >> 8)
         | ((val & 0x00ff) << 8);
}

inline int16_t bswap_16_h(const int16_t val) {
    const uint16_t v = bswap_16_h(*((uint16_t*)&val));
    return *((int16_t*)&v);
}

inline uint32_t bswap_32_h(uint32_t val) {
    return ((val & 0xff000000) >> 24)
         | ((val & 0x00ff0000) >>  8)
         | ((val & 0x0000ff00) <<  8)
         | ((val & 0x000000ff) << 24);
}

inline int32_t bswap_32_h(int32_t val) {
    uint32_t v = bswap_32_h(*((uint32_t*)&val));
    return *((int32_t*)&v);
}

inline float bswap_32_h(float val) {
    uint32_t v = bswap_32_h(*((uint32_t*)&val));
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

inline int64_t bswap_64_h(int64_t val) {
    uint64_t v = bswap_64_h(*((uint64_t*)&val));
    return *((int64_t*)&v);
}

inline double bswap_64_h(double val) {
    uint64_t v = bswap_64_h(*((uint64_t*)&val));
    return *((double*)&v);
}

inline uint32_t get_vrt_header_h(const VitaMetaData *meta) {
    return bswap_32_h(meta->vrt_header);
}

inline uint32_t get_stream_id_h(const VitaMetaData *meta) {
    return bswap_32_h(meta->stream_id);
}

inline uint32_t get_integer_time_h(const VitaMetaData *meta) {
    return bswap_32_h(meta->integer_time);
}

inline uint64_t get_fractional_time_h(const VitaMetaData *meta) {
    return bswap_64_h(meta->fractional_time);
}

inline uint32_t get_cif0_h(const ContextPacket *packet) {
    return bswap_32_h(packet->cif0);
}

inline double get_bandwidth_hz_h(const ContextPacket *packet) {
    return bswap_64_h(packet->bandwidth_hz) / (double)(__INT64_C(1) << BW_RADIX);
}

inline double get_rf_ref_freq_hz_h(const ContextPacket *packet) {
    return bswap_64_h(packet->rf_ref_freq_hz) / (double)(__INT64_C(1) << FREQ_RADIX);
}

inline float get_ref_level_dbm_h(const ContextPacket *packet) {
    return (int16_t)(bswap_32_h(packet->reference_level_dbm) & 0xFFFF) \
           / (float)(__INT16_C(1) << REF_LEVEL_RADIX);
}

inline float get_gain_1_db_h(const ContextPacket *packet) {
    return bswap_16_h(packet->gain_stage_1) / (float)(__INT16_C(1) << GAIN_RADIX);
}

inline float get_gain_2_db_h(const ContextPacket *packet) {
    return bswap_16_h(packet->gain_stage_2) / (float)(__INT16_C(1) << GAIN_RADIX);
}

inline double get_sample_rate_sps_h(const ContextPacket *packet) {
    return bswap_64_h(packet->sample_rate_sps) / (double)(__INT64_C(1) << SR_RADIX);
}

inline bool context_changed_h(const ContextPacket *packet) {
    return bswap_32_h(packet->cif0) & 0x80000000;
}
