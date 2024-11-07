// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "vrt_time.hpp"

VrtTime::VrtTime() {
    current_packet_time = std::chrono::steady_clock::now();
}

uint32_t VrtTime::intTime() const {
    auto since_epoch = current_packet_time.time_since_epoch();
    return std::chrono::duration_cast<std::chrono::seconds>(since_epoch).count();
}

uint64_t VrtTime::fracTime() const {
    auto since_epoch = current_packet_time.time_since_epoch();
    auto seconds = std::chrono::duration_cast<std::chrono::seconds>(since_epoch);
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(since_epoch - seconds);
    return nanoseconds.count();
}

void VrtTime::increment(uint32_t sample_count, double sample_rate) {
    std::chrono::duration<double> interval(static_cast<double>(sample_count) / sample_rate);
    current_packet_time += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
        interval);
}
