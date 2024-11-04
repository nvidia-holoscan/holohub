// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <chrono>

class VrtTime {
 public:
    VrtTime();
    void increment(uint32_t sample_count, double sample_rate);
    uint32_t intTime() const;
    uint64_t fracTime() const;

 private:
    std::chrono::time_point<std::chrono::steady_clock> current_packet_time;
};
