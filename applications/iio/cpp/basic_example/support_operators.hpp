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
#include "iio_params.hpp"

#include <iio.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

static constexpr int G_NUM_READS = 10;
static constexpr const char* G_URI = "ip:192.168.2.1";
static constexpr int G_NUM_CHANNELS = 2;  // Set to 1 or 2 to control number of channels

namespace holoscan::ops {

// =============================================================================
// Basic Test/Debug Operators
// =============================================================================

class BasicPrinterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicPrinterOp);
  BasicPrinterOp() = default;
  ~BasicPrinterOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<std::string>("value"); }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto value = op_input.receive<std::string>("value").value();
    HOLOSCAN_LOG_INFO("IIOAttributeRead value: {}", value);
  }
};

class BasicEmitterOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicEmitterOp);
  BasicEmitterOp() = default;
  ~BasicEmitterOp() = default;

  void setup(OperatorSpec& spec) override { spec.output<std::string>("value"); }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    auto value = std::string("manual");
    op_output.emit(value, "value");
  }
};

// =============================================================================
// IIO Buffer Test Operators
// =============================================================================

class BasicIIOBufferEmitterOP : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicIIOBufferEmitterOP);
  BasicIIOBufferEmitterOP() = default;
  ~BasicIIOBufferEmitterOP() = default;

  void setup(OperatorSpec& spec) override { spec.output<iio_buffer_info_t>("buffer"); }

  std::vector<int16_t> generateSineWave(ulong numSamples, float frequency, float amplitude,
                                        float sampleRate) {
    std::vector<int16_t> samples(numSamples);

    for (ulong i = 0; i < numSamples; ++i) {
      float t = i / sampleRate;
      samples[i] = amplitude * std::sin(2 * M_PI * frequency * t);
    }

    return samples;
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    // Test signal parameters
    constexpr uint enabled_channels = G_NUM_CHANNELS;
    constexpr ulong total_samples = 8192;                            // Total samples PER CHANNEL
    constexpr ulong buffer_size = total_samples * enabled_channels;  // Total buffer size
    constexpr float frequency = 8.0f;
    constexpr float amplitude = 408.0f;
    constexpr float sample_rate = 400.0f;

    // Device configuration (for test purposes)
    const std::string device_name = "cf-ad9361-dds-core-lpc";
    const std::string channel_name = "voltage0";
    const std::string channel_name2 = "voltage1";

    iio_context* ctx = iio_create_context_from_uri(G_URI);
    iio_device* dev = iio_context_find_device(ctx, device_name.c_str());
    iio_channel* chn = iio_device_find_channel(dev, channel_name.c_str(), true);
    iio_channel* chn2 = iio_device_find_channel(dev, channel_name2.c_str(), true);

    std::vector<int16_t> data_vector =
        generateSineWave(total_samples, frequency, amplitude, sample_rate);
    std::vector<int16_t> data_vector2 =
        generateSineWave(total_samples, frequency, amplitude / 2, sample_rate);

    // Create buffer info structure
    auto buffer_info = std::make_shared<iio_buffer_info_t>();
    buffer_info->buffer = new int16_t[buffer_size];
    buffer_info->is_cyclic = true;
    buffer_info->device_name = device_name;
    buffer_info->samples_count = total_samples;

    // Populate enabled channels using the helper function
    if (chn) {
      iio_channel_info_t ch1_info = create_channel_info_from_iio_channel(chn);
      buffer_info->enabled_channels.push_back(ch1_info);
    } else {
      // Fallback if channel not found
      iio_channel_info_t ch1_info;
      ch1_info.name = channel_name;
      ch1_info.is_output = true;
      ch1_info.index = 0;
      memset(&ch1_info.format, 0, sizeof(struct iio_data_format));
      buffer_info->enabled_channels.push_back(ch1_info);
    }

    if (enabled_channels == 2) {
      if (chn2) {
        iio_channel_info_t ch2_info = create_channel_info_from_iio_channel(chn2);
        buffer_info->enabled_channels.push_back(ch2_info);
      } else {
        // Fallback if channel not found
        iio_channel_info_t ch2_info;
        ch2_info.name = channel_name2;
        ch2_info.is_output = true;
        ch2_info.index = 1;
        memset(&ch2_info.format, 0, sizeof(struct iio_data_format));
        buffer_info->enabled_channels.push_back(ch2_info);
      }
    }

    // Fill buffer with interleaved samples for multi-channel setup
    auto* buffer = static_cast<int16_t*>(buffer_info->buffer);
    for (size_t sample_idx = 0; sample_idx < total_samples; ++sample_idx) {
      size_t buffer_idx = sample_idx * enabled_channels;

      // Channel 0
      buffer[buffer_idx] = data_vector[sample_idx];
      iio_channel_convert_inverse(chn, &buffer[buffer_idx], &buffer[buffer_idx]);

      // Channel 1 (if enabled)
      if (enabled_channels == 2) {
        buffer[buffer_idx + 1] = data_vector2[sample_idx];
        iio_channel_convert_inverse(chn2, &buffer[buffer_idx + 1], &buffer[buffer_idx + 1]);
      }
    }

    // Emit the buffer info
    op_output.emit(buffer_info, "buffer");
  }
};

class BasicIIOBufferPrinterOP : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicIIOBufferPrinterOP);
  BasicIIOBufferPrinterOP() = default;
  ~BasicIIOBufferPrinterOP() = default;

  void setup(OperatorSpec& spec) override { spec.input<iio_buffer_info_t>("buffer"); }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto buffer_info = op_input.receive<std::shared_ptr<iio_buffer_info_t>>("buffer").value();
    if (!buffer_info || !buffer_info->buffer) {
      HOLOSCAN_LOG_ERROR("Buffer is null");
      return;
    }

    const size_t enabled_channels = buffer_info->enabled_channels.size();

    // Print buffer metadata
    HOLOSCAN_LOG_INFO("Buffer info: samples_count={}, device={}, cyclic={}, channels={}",
                      buffer_info->samples_count,
                      buffer_info->device_name,
                      buffer_info->is_cyclic,
                      enabled_channels);

    // Print channel information
    for (const auto& ch : buffer_info->enabled_channels) {
      HOLOSCAN_LOG_INFO("  Channel: {} ({})", ch.name, ch.is_output ? "output" : "input");
    }

    // Print sample data
    printSampleData(buffer_info, static_cast<uint>(enabled_channels));
  }

 private:
  void printSampleData(std::shared_ptr<iio_buffer_info_t> buffer_info,
                       uint enabled_channels) const {
    constexpr size_t samples_to_print = 100;
    const auto* buffer = static_cast<int16_t*>(buffer_info->buffer);
    const size_t max_samples = std::min(samples_to_print, buffer_info->samples_count);

    HOLOSCAN_LOG_INFO("First {} samples per channel:", max_samples);

    if (enabled_channels == 1) {
      std::cout << "Channel 0: ";
      for (size_t i = 0; i < max_samples; ++i) {
        std::cout << buffer[i] << " ";
      }
      std::cout << std::endl;
    } else {
      // Print interleaved samples for each channel
      for (uint ch = 0; ch < enabled_channels; ++ch) {
        std::cout << "Channel " << ch << ": ";
        for (size_t i = 0; i < max_samples; ++i) {
          std::cout << buffer[i * enabled_channels + ch] << " ";
        }
        std::cout << std::endl;
      }
    }
  }
};

class BasicWaitOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BasicWaitOp);
  BasicWaitOp() = default;
  ~BasicWaitOp() = default;

  void setup(OperatorSpec&) override {}
  void compute(InputContext&, OutputContext&, ExecutionContext&) override { sleep(20); }
};

}  // namespace holoscan::ops
