// SPDX-FileCopyrightText: Copyright (c) 2026 Holoscan Team / NVIDIA. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Convert ARM Generic Timer ticks to nanoseconds.
//
// SIPL reports capture instants (frameCaptureTSC) and SIPLCaptureOp reads the live counter
// (cntvct_el0) in the same domain: raw ARM Generic Timer ticks, at whatever rate cntfrq_el0
// reports (31,250,000 Hz on Tegra/IGX Orin). Neither is nanoseconds, so both need this
// conversion before they can feed holoscan::sensor_io::ClockDiscipline, which primes and
// projects exclusively in nanoseconds.

#pragma once

#include <cstdint>

namespace holoscan::holoscan_camera {

// tsc * 1e9 / freq_hz, computed as sec*1e9 + rem*1e9/freq_hz (tsc = sec*freq_hz + rem) so the
// multiplication never approaches overflow the way tsc * 1'000'000'000 can for a tick count
// that has been running for hours. The two forms are exactly equal, not an approximation:
// floor(integer + x) equals integer + floor(x), and sec*1e9 has no fractional part to lose.
//
// freq_hz must be positive; the caller owns that guarantee (SIPLCaptureOp's cntfrq_el0 read
// falls back to a fixed constant rather than ever producing zero).
inline std::int64_t tsc_to_ns(std::uint64_t tsc, std::uint64_t freq_hz) noexcept {
  const std::uint64_t sec = tsc / freq_hz;
  const std::uint64_t rem = tsc % freq_hz;
  return static_cast<std::int64_t>(sec * 1'000'000'000ULL +
                                   rem * 1'000'000'000ULL / freq_hz);
}

}  // namespace holoscan::holoscan_camera
