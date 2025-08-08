/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Analog Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <holoscan/core/application.hpp>
#include <holoscan/core/conditions/gxf/boolean.hpp>
#include <holoscan/core/conditions/gxf/count.hpp>
#include <holoscan/core/endpoint.hpp>
#include <holoscan/core/forward_def.hpp>
#include <holoscan/core/operator.hpp>
#include "holoscan/holoscan.hpp"

#include "fft.hpp"
#include "iio_buffer_read.hpp"
#include "iio_buffer_write.hpp"
#include "iio_configurator.hpp"
#include "support_operators.hpp"

#include <filesystem>

using namespace holoscan;

class PlutoFFTExample : public holoscan::Application {
 public:
  explicit PlutoFFTExample(bool realtime = false) : realtime_(realtime) {
    name_ = realtime_ ? "PlutoFFTRealtimeExample" : "PlutoFFTExample";
  }

  void compose() override {
    // Common parameters
    const size_t fft_size = 16384;
    const size_t sample_rate_hz = 30'720'000;  // 30.72 MHz
    const int adc_bits = 12;                   // Pluto SDR ADC resolution

    // Derived parameters
    const size_t samples_per_channel = fft_size;                    // Number of I/Q pairs
    const size_t frequency_resolution = sample_rate_hz / fft_size;  // 1875 Hz

    // Window type: 2 = Hann (realtime mode prefers better sidelobes)
    const uint8_t window_type = 2;

    std::vector<std::string> enabled_channels_names = {"voltage0", "voltage1"};
    std::vector<bool> enabled_channels_output = {false, false};  // False for input channels
    const size_t num_channels = enabled_channels_names.size();

    // Start operator for dynamic flow control
    auto start_router_op = make_operator<ops::StartOp>("start_router");

    // Configuration path
    auto config_file_path = std::filesystem::path(__FILE__).parent_path() / "pluto_fft_config.yaml";
    auto iio_configurator_op =
        make_operator<ops::IIOConfigurator>("iio_configurator",
                                            Arg("cfg") = config_file_path.string(),
                                            make_condition<CountCondition>("config_count", 1));

    // Binary file reader - reads data from a binary file
    auto binary_file_path = std::filesystem::path(__FILE__).parent_path() / "sample_data.bin";
    std::shared_ptr<Operator> binary_reader_op;
    if (realtime_) {
      // For realtime mode, don't limit the count
      binary_reader_op = make_operator<ops::BinaryFileReaderOp>(
          "binary_reader",
          Arg("file_path") = binary_file_path.string(),
          Arg("device_name") = std::string("cf-ad9361-dds-core-lpc"),
          Arg("channel_names") = enabled_channels_names,
          Arg("channel_outputs") =
              std::vector<bool>{false, false},  // Input channels (voltage0, voltage1)
          Arg("samples_per_channel") = samples_per_channel,
          Arg("is_cyclic") = true,
          Arg("sample_size_bytes") = static_cast<size_t>(2));
    } else {
      // For single-shot mode, run only once
      binary_reader_op = make_operator<ops::BinaryFileReaderOp>(
          "binary_reader",
          Arg("file_path") = binary_file_path.string(),
          Arg("device_name") = std::string("cf-ad9361-dds-core-lpc"),
          Arg("channel_names") = enabled_channels_names,
          Arg("channel_outputs") =
              std::vector<bool>{false, false},  // Input channels (voltage0, voltage1)
          Arg("samples_per_channel") = samples_per_channel,
          Arg("is_cyclic") = true,
          Arg("sample_size_bytes") = static_cast<size_t>(2),
          make_condition<CountCondition>("binary_count", 1));
    }

    // IIOBufferWrite operator - writes data to Pluto SDR
    auto iio_buf_write_op = make_operator<ops::IIOBufferWrite>(
        "iio_buffer_write",
        Arg("ctx") = std::string(G_URI),
        Arg("dev") = std::string("cf-ad9361-dds-core-lpc"),
        Arg("is_cyclic") = true,
        Arg("enabled_channel_names") = enabled_channels_names,
        Arg("enabled_channel_output") = std::vector<bool>{true, true});  // Output channels

    // IIOBufferRead operator - reads data from Pluto SDR
    auto iio_buf_read_op =
        make_operator<ops::IIOBufferRead>("iio_buffer_read",
                                          Arg("ctx") = std::string(G_URI),
                                          Arg("dev") = std::string("cf-ad9361-lpc"),
                                          Arg("is_cyclic") = true,
                                          Arg("samples_count") = samples_per_channel,
                                          Arg("enabled_channel_names") = enabled_channels_names,
                                          Arg("enabled_channel_output") = enabled_channels_output);

    // IIOChannelConvertOp - skip conversion for binary file data
    auto iio_convert_op =
        make_operator<ops::IIOChannelConvertOp>("iio_convert", Arg("convert_channels") = false);

    // IIOBuffer2CudaTensorOp - converts IIO buffer to CUDA tensor
    auto buffer_to_tensor_op = make_operator<ops::IIOBuffer2CudaTensorOp>(
        "buffer_to_tensor",
        Arg("num_channels") = static_cast<unsigned int>(num_channels),
        Arg("samples_per_channel") = samples_per_channel,
        Arg("burst_size") = static_cast<int>(fft_size),
        Arg("num_bursts") = 1,
        Arg("adc_bits") = adc_bits);

    // FFT operator - performs FFT on the CUDA tensor
    auto fft_op =
        make_operator<ops::FFT>("fft",
                                Arg("burst_size") = static_cast<int>(fft_size),
                                Arg("num_bursts") = 1,
                                Arg("num_channels") = static_cast<uint16_t>(1),
                                Arg("spectrum_type") = static_cast<uint8_t>(0),
                                Arg("averaging_type") = static_cast<uint8_t>(0),
                                Arg("window_time") = static_cast<uint8_t>(0),
                                Arg("window_type") = window_type,
                                Arg("transform_points") = static_cast<uint32_t>(fft_size),
                                Arg("window_points") = static_cast<uint32_t>(fft_size),
                                Arg("resolution") = static_cast<uint64_t>(frequency_resolution),
                                Arg("span") = static_cast<uint64_t>(sample_rate_hz),
                                Arg("weighting_factor") = 1.0f,
                                Arg("f1_index") = static_cast<int32_t>(0),
                                Arg("f2_index") = static_cast<int32_t>(fft_size - 1),
                                Arg("window_time_delta") = static_cast<uint32_t>(1000));

    // Define common flows for dynamic routing from start operator
    add_flow(start_router_op, iio_configurator_op);
    add_flow(start_router_op, binary_reader_op);

    // Set dynamic flow control on the start operator
    set_dynamic_flows(start_router_op,
                      [iio_configurator_op, binary_reader_op, this](
                          const std::shared_ptr<holoscan::Operator>& op) mutable {
                        static bool config_done = false;
                        if (!config_done) {
                          HOLOSCAN_LOG_INFO("First run - routing to IIO configurator");
                          op->add_dynamic_flow(iio_configurator_op);
                          config_done = true;
                        } else {
                          HOLOSCAN_LOG_DEBUG("Configuration done - routing to binary file reader");
                          op->add_dynamic_flow(binary_reader_op);
                        }
                      });

    // Common data flow chain: binary_reader -> processing (bypass write/read cycle)
    add_flow(binary_reader_op, iio_convert_op, {{"buffer", "buffer_in"}});
    add_flow(iio_convert_op, buffer_to_tensor_op, {{"buffer_out", "buffer"}});
    add_flow(buffer_to_tensor_op, fft_op, {{"tensor", "in"}});

    // Create the appropriate visualization operator based on mode
    if (realtime_) {
      // FFTGnuplotRealtimeOp - real-time plotting with gnuplot
      auto fft_gnuplot_realtime_op = make_operator<ops::FFTGnuplotRealtimeOp>(
          "fft_gnuplot_realtime",
          Arg("max_frequency") = static_cast<float>(sample_rate_hz),
          Arg("power_offset") = 0.0f,  // Can be adjusted for calibration
          Arg("adc_bits") = adc_bits,
          Arg("update_interval") = 10,
          Arg("y_range") = std::vector<float>{-160.0f, 10.0f});  // dB range

      add_flow(fft_op, fft_gnuplot_realtime_op, {{"out", "buffer"}});
    } else {
      // FFTGnuplotOp - generates gnuplot visualization (runs once then stops)
      auto fft_gnuplot_op = make_operator<ops::FFTGnuplotOp>(
          "fft_gnuplot",
          Arg("output_file") = std::string("pluto_fft_spectrum"),
          Arg("selected_burst") = 0,
          Arg("max_frequency") = static_cast<float>(sample_rate_hz),
          Arg("power_offset") = 0.0f,  // Can be adjusted for calibration
          Arg("adc_bits") = adc_bits,
          make_condition<CountCondition>(1));  // Run only once, then application exits naturally

      add_flow(fft_op, fft_gnuplot_op, {{"out", "buffer"}});
    }
  }

 private:
  std::string name_;
  bool realtime_;
};

static int pluto_fft_main(int argc, char** argv) {
  auto app = holoscan::make_application<PlutoFFTExample>(false);
  app->run();

  return 0;
}

static int pluto_fft_realtime_main(int argc, char** argv) {
  auto app = holoscan::make_application<PlutoFFTExample>(true);
  app->run();

  return 0;
}
