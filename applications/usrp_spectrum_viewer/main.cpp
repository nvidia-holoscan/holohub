// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "usrp_rx/usrp_rx.hpp"
#include "uhd_chdr_rx/uhd_chdr_rx.hpp"
#include "log/log.hpp"
#include "spectrum_overlay/spectrum_overlay.hpp"
#include "spectrum_visualizer/spectrum_visualizer.hpp"
#include "spectrum_magnitude/spectrum_magnitude.hpp"
#include <fft.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <daqiri/daqiri.h>
#include <filesystem>
#include <memory>

class UsrpSpectrumViewerApp : public holoscan::Application {
 public:
    void compose() override {
        using namespace holoscan;

        // Shared state couples the Holoviz overlay controls with the spectrum
        // visualizer's pan/zoom and peak reporting logic.
        auto view_state = std::make_shared<ops::SpectrumViewState>();
        auto overlay_config = apps::SpectrumOverlayConfig{
            .db_min = from_config("spectrum_viz.db_min").as<float>(),
            .db_max = from_config("spectrum_viz.db_max").as<float>(),
            .initial_center_freq_mhz =
                from_config("spectrum_viz.ref_center_hz").as<float>() * 1.0e-6F,
            .initial_bandwidth_mhz =
                from_config("spectrum_viz.ref_bandwidth_hz").as<float>() * 1.0e-6F,
            .num_channels = from_config("spectrum_viz.num_channels").as<int>()};

        auto usrp_rx = make_operator<ops::UsrpRxOp>(
            "usrp_rx",
            from_config("usrp_rx"));

        auto uhd_chdr_rx = make_operator<ops::UhdChdrRxOp>(
            "uhd_chdr_rx",
            from_config("uhd_chdr_rx"));

        auto fft = make_operator<ops::FFT>(
            "fft",
            from_config("fft"));

        auto log = make_operator<ops::LogOp>(
            "log",
            from_config("log"),
            make_condition<CountCondition>(from_config("num_runs").as<int64_t>()));

        auto spectrum_magnitude = make_operator<ops::SpectrumMagnitudeOp>(
            "spectrum_magnitude",
            from_config("spectrum_magnitude"));

        // Managed stream pool lets the framework event-sync with Holoviz (no host block).
        auto spectrum_viz = make_operator<ops::SpectrumVisualizerOp>(
            "spectrum_viz",
            from_config("spectrum_viz"),
            make_resource<CudaStreamPool>("spectrum_viz_stream_pool", 0, 0, 0, 1, 5));
        // The visualizer reads live pan/zoom and peak-toggle state from the UI.
        spectrum_viz->set_view_state(view_state);

        auto holoviz = make_operator<ops::HolovizOp>(
            "holoviz",
            from_config("holoviz"),
            // The overlay callback owns the ImGui controls and writes updated
            // display parameters back into the shared view state.
            Arg("layer_callback",
                apps::make_spectrum_overlay_callback(overlay_config, view_state)),
            Arg("cuda_stream_pool",
                make_resource<CudaStreamPool>("viz_stream_pool", 0, 0, 0, 1, 5)));

        holoviz->metadata_policy(holoscan::MetadataPolicy::kUpdate);

        // UsrpRxOp is control-only (started by start_op(), emits no tensors);
        // UhdChdrRxOp is the real source, fed out-of-band over the NIC (DPDK).
        add_flow(start_op(), usrp_rx);
        add_flow(uhd_chdr_rx, fft);
        add_flow(fft, log);
        add_flow(fft, spectrum_magnitude);
        add_flow(spectrum_magnitude, spectrum_viz);
        add_flow(spectrum_viz, holoviz, {{"outputs", "receivers"}});
        add_flow(spectrum_viz, holoviz, {{"output_specs", "input_specs"}});
    }
};

int main(int argc, char** argv) {
    // Create the application
    auto app = holoscan::make_application<UsrpSpectrumViewerApp>();

    // Enable metadata for all operators
    app->enable_metadata(true);

    // Check for required configuration file argument
    if (argc < 2) {
        HOLOSCAN_LOG_ERROR("Usage: {} [config.yaml]", argv[0]);
        return -1;
    }

    // Get the full path to the configuration file
    std::filesystem::path config_path(argv[1]);
    if (!config_path.is_absolute()) {
        config_path = std::filesystem::canonical("/proc/self/exe").parent_path() / config_path;
    }

    // Check if the configuration file exists
    if (!std::filesystem::exists(config_path)) {
        HOLOSCAN_LOG_ERROR("Configuration file '{}' does not exist",
                static_cast<std::string>(config_path));
        return -1;
    }

    if (daqiri::daqiri_init(config_path.string()) != daqiri::Status::SUCCESS) {
        HOLOSCAN_LOG_ERROR("Failed to configure DAQIRI");
        return -1;
    }
    HOLOSCAN_LOG_INFO("Configured DAQIRI");

    // Apply configuration from file
    app->config(config_path);

    // Configure the event-based scheduler
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
          "event-based-scheduler", app->from_config("scheduler")));

    // Run the application
    app->run();

    daqiri::print_stats();
    daqiri::shutdown();
    return 0;
}
