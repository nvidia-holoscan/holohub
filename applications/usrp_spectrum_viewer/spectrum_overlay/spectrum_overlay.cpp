// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0

#include "spectrum_overlay.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

#include <imgui.h>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <holoviz/holoviz.hpp>

namespace holoscan::apps {

namespace {

constexpr float MIN_BANDWIDTH_MHZ = 0.001F;

struct SpectrumOverlayUiState {
  explicit SpectrumOverlayUiState(SpectrumOverlayConfig config)
      : reset_center_freq_mhz(config.initial_center_freq_mhz),
        reset_bandwidth_mhz(config.initial_bandwidth_mhz),
        db_min(config.db_min),
        db_max(config.db_max),
        num_channels(config.num_channels),
        center_freq_mhz(config.initial_center_freq_mhz),
        bandwidth_mhz(config.initial_bandwidth_mhz) {}

  const float reset_center_freq_mhz;
  const float reset_bandwidth_mhz;
  const float db_min;
  const float db_max;
  const int num_channels;
  float center_freq_mhz;
  float bandwidth_mhz;
  bool show_peak = false;
};

}  // namespace

ops::HolovizOp::LayerCallbackFunction make_spectrum_overlay_callback(
    SpectrumOverlayConfig config,
    std::shared_ptr<holoscan::ops::SpectrumViewState> view_state) {
  if (!std::isfinite(config.initial_bandwidth_mhz)
      || config.initial_bandwidth_mhz < MIN_BANDWIDTH_MHZ) {
    config.initial_bandwidth_mhz = MIN_BANDWIDTH_MHZ;
  }
  // Seed the shared state before the first frame so SpectrumVisualizerOp starts
  // with the same center/span the UI will present to the user.
  if (view_state) {
    view_state->store_view(config.initial_center_freq_mhz, config.initial_bandwidth_mhz);
  }

  auto ui_state = std::make_shared<SpectrumOverlayUiState>(config);

  return [ui_state = std::move(ui_state), view_state = std::move(view_state)](
             const std::vector<holoscan::gxf::Entity>&) {
    using namespace holoscan;

    // Draw the interactive control panel first so the latest values can be
    // pushed back into shared state before the next visualizer compute.
    viz::BeginImGuiLayer();
    ImGui::Begin("Spectrum Controls", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::InputFloat("Center Freq (MHz)", &ui_state->center_freq_mhz, 1.0F, 10.0F, "%.1f");
    ImGui::InputFloat("Bandwidth (MHz)", &ui_state->bandwidth_mhz, 1.0F, 10.0F, "%.1f");
    // A non-positive span makes the frequency mapping undefined; keep it above
    // a small positive floor so labels and peak markers stay finite.
    if (ImGui::Button("Reset")) {
      ui_state->center_freq_mhz = ui_state->reset_center_freq_mhz;
      ui_state->bandwidth_mhz = ui_state->reset_bandwidth_mhz;
    }
    if (!std::isfinite(ui_state->center_freq_mhz)) {
      ui_state->center_freq_mhz = ui_state->reset_center_freq_mhz;
    }
    if (!std::isfinite(ui_state->bandwidth_mhz)
        || ui_state->bandwidth_mhz < MIN_BANDWIDTH_MHZ) {
      ui_state->bandwidth_mhz = MIN_BANDWIDTH_MHZ;
    }
    ImGui::Text("Span: %.1f - %.1f MHz",
                ui_state->center_freq_mhz - ui_state->bandwidth_mhz / 2.0F,
                ui_state->center_freq_mhz + ui_state->bandwidth_mhz / 2.0F);

    // Channel color legend: one swatch per channel using the same palette the
    // visualizer applies to each LINE_STRIP, so users can match trace to channel.
    ImGui::Separator();
    ImGui::Text("Channels:");
    const int palette_size = static_cast<int>(ops::CHANNEL_PALETTE.size());
    for (int ch = 0; ch < ui_state->num_channels; ++ch) {
      const auto& color = ops::CHANNEL_PALETTE[ch % palette_size];
      char swatch_id[16];
      std::snprintf(swatch_id, sizeof(swatch_id), "##ch%d", ch);
      ImGui::ColorButton(swatch_id,
                         ImVec4(color[0], color[1], color[2], color[3]),
                         ImGuiColorEditFlags_NoTooltip | ImGuiColorEditFlags_NoPicker,
                         ImVec2(14.0F, 14.0F));
      ImGui::SameLine();
      ImGui::Text("CH%d", ch);
    }

    if (ImGui::Button(ui_state->show_peak ? "Hide Peak" : "Show Peak")) {
      ui_state->show_peak = !ui_state->show_peak;
    }
    if (ui_state->show_peak && view_state) {
      const int peak_palette_size = static_cast<int>(ops::CHANNEL_PALETTE.size());
      const int shown_channels =
          std::min(ui_state->num_channels, static_cast<int>(ops::MAX_CHANNELS));
      for (int ch = 0; ch < shown_channels; ++ch) {
        float peak_freq_mhz = 0.0F;
        float peak_db_val = 0.0F;
        view_state->load_peak(ch, peak_freq_mhz, peak_db_val);
        const auto& color = ops::CHANNEL_PALETTE[ch % peak_palette_size];
        ImGui::TextColored(ImVec4(color[0], color[1], color[2], color[3]),
                           "CH%d Peak: %.3f MHz  @  %.1f dB", ch, peak_freq_mhz, peak_db_val);
      }
    }
    ImGui::End();
    viz::EndLayer();

    if (view_state) {
      view_state->store_view(ui_state->center_freq_mhz, ui_state->bandwidth_mhz);
      view_state->show_peak.store(ui_state->show_peak, std::memory_order_relaxed);
    }

    // Draw axes and annotations in a separate geometry layer so they appear on
    // top of the rendered spectrum lines.
    viz::BeginGeometryLayer();

    // Grid/label divisions per axis, shared by the grid geometry and the tick
    // labels so they always line up. The frequency (x) axis uses a fixed number
    // of divisions; the power (y) axis uses one division per db_per_division dB.
    constexpr int frequency_divisions = 10;
    constexpr float db_per_division = 10.0F;
    const int power_divisions = std::max(
        1,
        static_cast<int>(std::round((ui_state->db_max - ui_state->db_min) / db_per_division)));

    {
      const int total_grid_lines = (frequency_divisions + 1) + (power_divisions + 1);
      std::vector<float> grid_data;
      grid_data.reserve(static_cast<size_t>(total_grid_lines) * 4);
      const float gx0 = ops::PLOT_LEFT;
      const float gx1 = ops::PLOT_LEFT + ops::PLOT_WIDTH;
      const float gy0 = ops::PLOT_TOP;
      const float gy1 = ops::PLOT_TOP + ops::PLOT_HEIGHT;

      // Horizontal grid lines, one per power (dB) division.
      for (int index = 0; index <= power_divisions; ++index) {
        const float y = gy0 + (static_cast<float>(index) / power_divisions) * (gy1 - gy0);
        grid_data.insert(grid_data.end(), {gx0, y, gx1, y});
      }
      // Vertical grid lines, one per frequency division.
      for (int index = 0; index <= frequency_divisions; ++index) {
        const float x = gx0 + (static_cast<float>(index) / frequency_divisions) * ops::PLOT_WIDTH;
        grid_data.insert(grid_data.end(), {x, gy0, x, gy1});
      }

      viz::Color(0.35F, 0.35F, 0.35F, 0.6F);
      viz::LineWidth(1.0F);
      viz::Primitive(viz::PrimitiveTopology::LINE_LIST,
                     total_grid_lines,
                     grid_data.size(),
                     grid_data.data());
    }

    viz::Color(1.0F, 1.0F, 1.0F, 1.0F);
    viz::Text(0.47F, 0.975F, 0.022F, "Frequency (MHz)");
    viz::Text(0.005F, 0.015F, 0.022F, "Power (dBFS)");

    const float f_lo = ui_state->center_freq_mhz - ui_state->bandwidth_mhz / 2.0F;
    const float f_hi = ui_state->center_freq_mhz + ui_state->bandwidth_mhz / 2.0F;
    char buffer[32];
    // Label the x-axis using the same frequency divisions as the vertical grid.
    for (int index = 0; index <= frequency_divisions; ++index) {
      const float frac = static_cast<float>(index) / static_cast<float>(frequency_divisions);
      const float freq = f_lo + frac * (f_hi - f_lo);
      std::snprintf(buffer, sizeof(buffer), "%.0f", freq);
      viz::Text(ops::PLOT_LEFT + frac * ops::PLOT_WIDTH, 0.955F, 0.016F, buffer);
    }

    // Label the y-axis top to bottom to match the visualizer dB normalization,
    // using the same power divisions as the horizontal grid.
    for (int index = 0; index <= power_divisions; ++index) {
      const float frac = static_cast<float>(index) / static_cast<float>(power_divisions);
      const float db = ui_state->db_max + frac * (ui_state->db_min - ui_state->db_max);
      std::snprintf(buffer, sizeof(buffer), "%.0f", db);
      viz::Text(0.005F, ops::PLOT_TOP + frac * ops::PLOT_HEIGHT, 0.016F, buffer);
    }

    if (ui_state->show_peak && view_state) {
      // Convert each channel's reported peak back into overlay coordinates so the
      // annotation follows the current center frequency, span, and dB limits.
      const int peak_palette_size = static_cast<int>(ops::CHANNEL_PALETTE.size());
      const int shown_channels =
          std::min(ui_state->num_channels, static_cast<int>(ops::MAX_CHANNELS));
      for (int ch = 0; ch < shown_channels; ++ch) {
        float peak_freq_mhz = 0.0F;
        float peak_db_val = 0.0F;
        view_state->load_peak(ch, peak_freq_mhz, peak_db_val);
        if (!std::isfinite(peak_freq_mhz) || !std::isfinite(peak_db_val)) {
          continue;
        }
        const float u = (peak_freq_mhz - ui_state->center_freq_mhz) / ui_state->bandwidth_mhz
                        + ops::PLOT_CENTER_U;

        if (u < 0.0F || u > 1.0F) {
          continue;
        }
        const float px = ops::PLOT_LEFT + ops::PLOT_WIDTH * u;
        const float norm = (peak_db_val - ui_state->db_min) / (ui_state->db_max - ui_state->db_min);
        float py = ops::PLOT_TOP + ops::PLOT_HEIGHT * (1.0F - norm);
        py = std::clamp(py, ops::PLOT_TOP, ops::PLOT_TOP + ops::PLOT_HEIGHT);
        std::snprintf(buffer, sizeof(buffer), "CH%d %.2f MHz, %.1f dB",
                      ch, peak_freq_mhz, peak_db_val);
        const auto& color = ops::CHANNEL_PALETTE[ch % peak_palette_size];
        viz::Color(color[0], color[1], color[2], 1.0F);
        viz::Text(px, std::max(ops::PLOT_TOP, py - 0.03F), 0.016F, buffer);
      }
    }

    viz::EndLayer();
  };
}

}  // namespace holoscan::apps
