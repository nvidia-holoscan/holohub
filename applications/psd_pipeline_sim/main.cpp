// SPDX-FileCopyrightText: 2024 Valley Tech Systems, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <file_reader.hpp>
#include <fft.hpp>
#include <high_rate_psd.hpp>
#include <low_rate_psd.hpp>
#include <vita49_psd_packetizer.hpp>

class PsdPipelineSim : public holoscan::Application {
 public:
    void compose() override {
        using namespace holoscan;

        auto file_reader_rate_hz = std::to_string(
            from_config("file_reader.sample_rate_sps").as<uint64_t>()
            / from_config("file_reader.burst_size").as<uint64_t>())
            + std::string("Hz");

        auto fileReaderOp = make_operator<ops::FileReader>(
            "fileReader",
            from_config("file_reader"),
            make_condition<PeriodicCondition>("periodic-condition",
                                              Arg("recess_period") = file_reader_rate_hz));
        auto fftOp = make_operator<ops::FFT>(
            "fftOp",
            from_config("fft"));
        auto highRatePsdOp = make_operator<ops::HighRatePSD>(
            "highRatePsdOp",
            from_config("high_rate_psd"));
        auto lowRatePsdOp = make_operator<ops::LowRatePSD>(
            "lowRatePsdOp",
            from_config("low_rate_psd"));
        auto psdPacketizer = make_operator<ops::V49PsdPacketizer>(
            "psdPacketizer",
            from_config("vita49_psd_packetizer"),
            make_condition<CountCondition>(from_config("num_psds").as<int64_t>()));
        add_operator(fileReaderOp);
        add_operator(fftOp);
        add_operator(psdPacketizer);
        add_flow(fileReaderOp, fftOp);
        add_flow(fftOp, highRatePsdOp);
        add_flow(highRatePsdOp, lowRatePsdOp);
        add_flow(lowRatePsdOp, psdPacketizer);
    }
};

int main(int argc, char** argv) {
    holoscan::set_log_pattern("FULL");
    auto app = holoscan::make_application<PsdPipelineSim>();

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
    app->run();
    return 0;
}
