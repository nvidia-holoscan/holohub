// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>

#include <holoscan/holoscan.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>

#include "spectrum_visualizer/spectrum_view_state.hpp"

namespace holoscan::apps {

struct SpectrumOverlayConfig {
  float db_min = -120.0F;
  float db_max = 0.0F;
  float initial_center_freq_mhz = 1000.0F;
  float initial_bandwidth_mhz = 500.0F;
  int num_channels = 1;
};

// Build the Holoviz layer callback used by the application. The callback keeps
// its own UI-local state but exchanges pan/zoom and peak values through
// SpectrumViewState.
ops::HolovizOp::LayerCallbackFunction make_spectrum_overlay_callback(
    SpectrumOverlayConfig config,
    std::shared_ptr<holoscan::ops::SpectrumViewState> view_state);

}  // namespace holoscan::apps
