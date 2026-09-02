// SPDX-FileCopyrightText: 2026 National Instruments Corporation
//
// SPDX-License-Identifier: Apache-2.0
#include "usrp_rx/usrp_rx.hpp"
#include "uhd_chdr_rx/uhd_chdr_rx.hpp"
#include "log/log.hpp"
#include "spectrum_overlay/spectrum_overlay.hpp"
#include "spectrum_visualizer/spectrum_visualizer.hpp"
#include "spectrum_magnitude/spectrum_magnitude.hpp"

#include <cstdint>
#include <exception>
#include <fft.hpp>
#include <holoscan/operators/holoviz/holoviz.hpp>
#include <daqiri/daqiri.h>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

namespace {

class DaqiriSession {
 public:
    DaqiriSession() = default;
    DaqiriSession(const DaqiriSession&) = delete;
    DaqiriSession& operator=(const DaqiriSession&) = delete;

    ~DaqiriSession() noexcept {
        try {
            daqiri::print_stats();
        } catch (const std::exception& error) {
            HOLOSCAN_LOG_ERROR("Failed to print DAQIRI statistics: {}", error.what());
        } catch (...) {
            HOLOSCAN_LOG_ERROR("Failed to print DAQIRI statistics");
        }

        try {
            daqiri::shutdown();
        } catch (const std::exception& error) {
            HOLOSCAN_LOG_ERROR("Failed to shut down DAQIRI: {}", error.what());
        } catch (...) {
            HOLOSCAN_LOG_ERROR("Failed to shut down DAQIRI");
        }
    }
};

}  // namespace

class UsrpSpectrumViewerApp : public holoscan::Application {
 public:
    void compose() override {
        using namespace holoscan;

        const int64_t rx_channels = from_config("uhd_chdr_rx.num_channels").as<int64_t>();
        const int64_t fft_channels = from_config("fft.num_channels").as<int64_t>();
        const int64_t magnitude_channels =
            from_config("spectrum_magnitude.num_channels").as<int64_t>();
        const int64_t viz_channels = from_config("spectrum_viz.num_channels").as<int64_t>();
        const int64_t log_channels = from_config("log.num_channels").as<int64_t>();
        if (rx_channels <= 0 || rx_channels != fft_channels
            || rx_channels != magnitude_channels || rx_channels != viz_channels
            || rx_channels != log_channels) {
            throw std::runtime_error(fmt::format(
                "Pipeline num_channels must be positive and equal: rx={}, fft={}, magnitude={}, "
                "viz={}, log={}",
                rx_channels,
                fft_channels,
                magnitude_channels,
                viz_channels,
                log_channels));
        }

        const bool usrp_enabled = from_config("usrp_rx.enabled").as<bool>();
        if (usrp_enabled) {
            const auto usrp_channels =
                from_config("usrp_rx.channels").as<std::vector<int64_t>>();
            if (static_cast<int64_t>(usrp_channels.size()) != rx_channels) {
                throw std::runtime_error(fmt::format(
                    "usrp_rx.channels has {} entries but the pipeline is configured for {} "
                    "channels",
                    usrp_channels.size(),
                    rx_channels));
            }
        }

        const int64_t samples_per_packet =
            from_config("uhd_chdr_rx.num_complex_samples_per_packet").as<int64_t>();
        const int64_t packets_per_output =
            from_config("uhd_chdr_rx.num_packets_per_output").as<int64_t>();
        const int64_t outputs_per_batch =
            from_config("uhd_chdr_rx.num_outputs_per_batch").as<int64_t>();
        const int64_t fft_burst_size = from_config("fft.burst_size").as<int64_t>();
        const int64_t fft_bursts = from_config("fft.num_bursts").as<int64_t>();
        const int64_t transform_points = from_config("fft.transform_points").as<int64_t>();
        const int64_t window_points = from_config("fft.window_points").as<int64_t>();
        const int64_t magnitude_burst_size =
            from_config("spectrum_magnitude.burst_size").as<int64_t>();
        const int64_t magnitude_bursts =
            from_config("spectrum_magnitude.num_bursts").as<int64_t>();
        const int64_t magnitude_averages =
            from_config("spectrum_magnitude.num_averages").as<int64_t>();
        const int64_t viz_bins = from_config("spectrum_viz.num_bins").as<int64_t>();
        if (samples_per_packet <= 0 || packets_per_output <= 0 || outputs_per_batch <= 0) {
            throw std::runtime_error("UHD CHDR RX batching dimensions must all be > 0");
        }
        if (usrp_enabled
            && from_config("usrp_rx.spp").as<int64_t>() != samples_per_packet) {
            throw std::runtime_error(fmt::format(
                "usrp_rx.spp must equal uhd_chdr_rx.num_complex_samples_per_packet ({})",
                samples_per_packet));
        }
        const int64_t samples_per_output = samples_per_packet * packets_per_output;
        if (samples_per_output != fft_burst_size
            || samples_per_output != transform_points
            || samples_per_output != window_points
            || samples_per_output != magnitude_burst_size
            || samples_per_output != viz_bins) {
            throw std::runtime_error(fmt::format(
                "Samples per output ({}) must equal fft.burst_size/transform_points/"
                "window_points, spectrum_magnitude.burst_size, and spectrum_viz.num_bins",
                samples_per_output));
        }
        if (outputs_per_batch != fft_bursts || outputs_per_batch != magnitude_bursts
            || outputs_per_batch != magnitude_averages) {
            throw std::runtime_error(fmt::format(
                "uhd_chdr_rx.num_outputs_per_batch ({}) must equal fft.num_bursts, "
                "spectrum_magnitude.num_bursts, and spectrum_magnitude.num_averages",
                outputs_per_batch));
        }

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
    DaqiriSession daqiri_session;

    // Create the application after the DAQIRI session guard so application
    // resources are released before DAQIRI on every return or exception path.
    auto app = holoscan::make_application<UsrpSpectrumViewerApp>();
    app->enable_metadata(true);

    // Apply configuration from file
    app->config(config_path);

    // Configure the event-based scheduler
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
          "event-based-scheduler", app->from_config("scheduler")));

    // Run the application
    app->run();

    return 0;
}
