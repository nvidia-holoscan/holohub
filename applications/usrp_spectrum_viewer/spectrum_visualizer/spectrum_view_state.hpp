// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <cstring>

namespace holoscan::ops {

// Distinct line colors (RGBA) for up to four channels, shared by the visualizer
// (line geometry) and the overlay (control-window legend) so both stay in sync.
inline constexpr std::array<std::array<float, 4>, 4> CHANNEL_PALETTE{{
    {{0.1F, 0.3F, 1.0F, 0.9F}},    // blue
    {{1.0F, 0.2F, 0.2F, 0.9F}},    // red
    {{1.0F, 0.6F, 0.2F, 0.9F}},    // orange
    {{1.0F, 0.3F, 0.8F, 0.9F}}}};  // magenta

// Maximum number of channels the shared state and palette can distinctly track.
inline constexpr size_t MAX_CHANNELS = CHANNEL_PALETTE.size();

// Plot area as a fraction of the window, shared by the visualizer geometry and
// the axis overlay so the spectrum line, grid, and tick labels stay aligned
// while panning/zooming. (left, top) is the top-left corner; width/height are
// the size. Screen y grows downward, so PLOT_TOP maps to db_max (top of plot).
inline constexpr float PLOT_LEFT = 0.02F;    // left edge (x); right = PLOT_LEFT + PLOT_WIDTH
inline constexpr float PLOT_WIDTH = 0.94F;
inline constexpr float PLOT_TOP = 0.05F;     // top edge (y); bottom = PLOT_TOP + PLOT_HEIGHT
inline constexpr float PLOT_HEIGHT = 0.88F;

// Plot center on the normalized [0, 1] x axis.
inline constexpr float PLOT_CENTER_U = 0.5F;

// Center-to-edge fraction of the bandwidth.
inline constexpr float HALF_BANDWIDTH_FRACTION = 0.5F;

/**
 * Lock-free view parameters shared between the UI overlay (writer) and the
 * SpectrumVisualizer (reader), used to pan/zoom the rendered spectrum.
 */
struct SpectrumViewState {
  // center_freq_mhz and bandwidth_mhz packed into one 64-bit atomic (center in
  // the high 32 bits) so the visualizer always reads the pair without a torn read.
  std::atomic<uint64_t> view_params{pack_view(1000.0F, 500.0F)};

  // Per-channel detected peak: absolute frequency (MHz) and the power (dB) at
  // that bin. The visualizer writes one slot per channel; the UI overlay reads
  // them to annotate each channel independently.
  std::array<std::atomic<float>, MAX_CHANNELS> peak_freq_mhz{};
  std::array<std::atomic<float>, MAX_CHANNELS> peak_db{};
  // Set by the UI overlay when the user toggles "Show Peak". When false, the
  // visualizer skips the per-frame D2H copy + CPU argmax used for peak finding.
  std::atomic<bool> show_peak{false};

  SpectrumViewState() {
    for (auto& db : peak_db) {
      db.store(-200.0F, std::memory_order_relaxed);
    }
  }

  // Atomically publish the current center frequency and bandwidth (MHz).
  void store_view(float center_freq_mhz, float bandwidth_mhz) {
    view_params.store(pack_view(center_freq_mhz, bandwidth_mhz), std::memory_order_relaxed);
  }

  // Atomically read the current center frequency and bandwidth (MHz) as a pair.
  void load_view(float& center_freq_mhz, float& bandwidth_mhz) const {
    unpack_view(view_params.load(std::memory_order_relaxed), center_freq_mhz, bandwidth_mhz);
  }

  static uint64_t pack_view(float center_freq_mhz, float bandwidth_mhz) {
    uint32_t center_bits = 0;
    uint32_t bandwidth_bits = 0;
    std::memcpy(&center_bits, &center_freq_mhz, sizeof(center_bits));
    std::memcpy(&bandwidth_bits, &bandwidth_mhz, sizeof(bandwidth_bits));
    return (static_cast<uint64_t>(center_bits) << 32) | static_cast<uint64_t>(bandwidth_bits);
  }

  static void unpack_view(uint64_t packed, float& center_freq_mhz, float& bandwidth_mhz) {
    const uint32_t center_bits = static_cast<uint32_t>(packed >> 32);
    const uint32_t bandwidth_bits = static_cast<uint32_t>(packed & 0xFFFFFFFFULL);
    std::memcpy(&center_freq_mhz, &center_bits, sizeof(center_freq_mhz));
    std::memcpy(&bandwidth_mhz, &bandwidth_bits, sizeof(bandwidth_mhz));
  }
};

}  // namespace holoscan::ops
