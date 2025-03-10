// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "advanced_network_connectors/vita49_rx.h"
#include <fft.hpp>
#include <high_rate_psd.hpp>
#include <low_rate_psd.hpp>
#include <vita49_psd_packetizer.hpp>
#include <data_writer.hpp>

// #define WRITE_DATA

class PsdPipeline : public holoscan::Application {
 public:
    void compose() override {
        using namespace holoscan;

        auto advNetOp = make_operator<ops::AdvNetworkOpRx>(
            "advNetOp",
            from_config("advanced_network"),
            make_condition<BooleanCondition>("is_alive", true));

        auto vitaConnectorOp = make_operator<ops::Vita49ConnectorOpRx>(
            "vitaConnectorOp",
            from_config("vita_connector"));

        auto fftOp = make_operator<ops::FFT>(
            "fftOp",
            from_config("fft"));

        auto highRatePsdOp = make_operator<ops::HighRatePSD>(
            "highRatePsdOp",
            from_config("high_rate_psd"));

        auto lowRatePsdOp = make_operator<ops::LowRatePSD>(
            "lowRatePsdOp",
            from_config("low_rate_psd"));

        auto packetizerOp = make_operator<ops::V49PsdPacketizer>(
            "packetizerOp",
            from_config("vita49_psd_packetizer"),
            make_condition<CountCondition>(from_config("num_psds").as<int64_t>()));

        add_operator(advNetOp);
        add_operator(vitaConnectorOp);
        add_operator(fftOp);
        add_operator(highRatePsdOp);
        add_operator(lowRatePsdOp);
        add_operator(packetizerOp);
        add_flow(advNetOp, vitaConnectorOp, {{"bench_rx_out", "in"}});
        add_flow(vitaConnectorOp, fftOp);
        add_flow(fftOp, highRatePsdOp);
        add_flow(highRatePsdOp, lowRatePsdOp);
        add_flow(lowRatePsdOp, packetizerOp);

#ifdef WRITE_DATA
        auto dataWriterOp = make_operator<ops::DataWriter>(
            "dataWriterOp",
            from_config("data_writer"),
            make_condition<CountCondition>(2));
        add_operator(dataWriterOp);
        add_flow(vitaConnectorOp, dataWriterOp);
#endif
    }
};

int main(int argc, char** argv) {
    holoscan::set_log_pattern("FULL");
    auto app = holoscan::make_application<PsdPipeline>();

    // Get the configuration file
    if (argc < 1) {
        HOLOSCAN_LOG_ERROR("Usage: {} [config.yaml]", argv[0]);
        return -1;
    }

    auto config_path = std::filesystem::canonical(argv[0]).parent_path();
    config_path += "/" + std::string(argv[1]);

    // Check if the file exists
    if (!std::filesystem::exists(config_path)) {
        HOLOSCAN_LOG_ERROR("Configuration file '{}' does not exist",
                static_cast<std::string>(config_path));
        return -1;
    }

    // Run
    app->is_metadata_enabled(true);
    app->config(config_path);
    app->scheduler(app->make_scheduler<holoscan::EventBasedScheduler>(
          "event-based-scheduler", app->from_config("scheduler")));
    app->run();
    return 0;
}
