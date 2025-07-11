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

#include <cuda_runtime.h>
#include <iio.h>
#include <matx.h>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <vector>

using namespace matx;
using complex = cuda::std::complex<float>;

static constexpr const char* G_URI = "ip:192.168.2.1";

namespace holoscan::ops {

// =============================================================================
// Binary File Reader Operator
// =============================================================================

class BinaryFileReaderOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(BinaryFileReaderOp);
  BinaryFileReaderOp() = default;
  ~BinaryFileReaderOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.output<std::shared_ptr<iio_buffer_info_t>>("buffer");

    spec.param(file_path_, "file_path", "File Path", "Path to the binary file to read");
    spec.param(device_name_, "device_name", "Device Name", "Name of the target device");
    spec.param(channel_names_, "channel_names", "Channel Names", "List of channel names");
    spec.param(
        channel_outputs_, "channel_outputs", "Channel Outputs", "List of channel output flags");
    spec.param(samples_per_channel_,
               "samples_per_channel",
               "Samples Per Channel",
               "Number of samples per channel");
    spec.param(is_cyclic_, "is_cyclic", "Is Cyclic", "Whether the buffer should be cyclic", true);
    spec.param(sample_size_bytes_,
               "sample_size_bytes",
               "Sample Size",
               "Size of each sample in bytes",
               static_cast<size_t>(2));
  }

  void initialize() override {
    Operator::initialize();

    // Initialize file offset for sliding window
    current_offset_ = 0;

    // Open file once to get size
    std::ifstream file(file_path_.get(), std::ios::binary | std::ios::ate);
    if (file.is_open()) {
      file_size_ = file.tellg();
      file.close();
    }
  }

  void compute(InputContext&, OutputContext& op_output, ExecutionContext&) override {
    // Calculate expected size based on parameters
    size_t num_channels = channel_names_.get().size();
    size_t expected_size = samples_per_channel_.get() * num_channels * sample_size_bytes_.get();

    // Check if we need to wrap around to the beginning of the file
    if (current_offset_ + expected_size > file_size_) {
      current_offset_ = 0;
      HOLOSCAN_LOG_DEBUG("Wrapping around to beginning of file");
    }

    // Read binary file at current offset
    std::ifstream file(file_path_.get(), std::ios::binary);
    if (!file.is_open()) {
      HOLOSCAN_LOG_ERROR("Failed to open file: {}", file_path_.get());
      return;
    }

    // Seek to current offset
    file.seekg(current_offset_);

    // Create buffer info
    auto buffer_info = std::make_shared<iio_buffer_info_t>();
    buffer_info->samples_count = samples_per_channel_.get();
    buffer_info->is_cyclic = is_cyclic_.get();
    buffer_info->device_name = device_name_.get();

    // Allocate buffer and read data
    buffer_info->buffer = new uint8_t[expected_size];
    file.read(reinterpret_cast<char*>(buffer_info->buffer), expected_size);

    if (!file) {
      HOLOSCAN_LOG_ERROR("Failed to read {} bytes from offset {}", expected_size, current_offset_);
      delete[] static_cast<uint8_t*>(buffer_info->buffer);
      file.close();
      return;
    }

    file.close();

    // Update offset for next read (slide by half the window for overlap)
    size_t slide_amount = expected_size / 2;  // 50% overlap
    current_offset_ += slide_amount;

    HOLOSCAN_LOG_DEBUG("Read {} bytes from offset {}, next offset: {}",
                       expected_size,
                       current_offset_ - slide_amount,
                       current_offset_);

    // Populate channel information
    auto& channel_names = channel_names_.get();
    auto& channel_outputs = channel_outputs_.get();

    for (size_t i = 0; i < channel_names.size(); ++i) {
      iio_channel_info_t chan_info{};
      chan_info.name = channel_names[i];
      chan_info.is_output = (i < channel_outputs.size()) ? channel_outputs[i] : true;
      chan_info.index = static_cast<unsigned int>(i);
      // Format is not needed since we're not using IIO channel conversion

      buffer_info->enabled_channels.push_back(chan_info);
    }

    HOLOSCAN_LOG_DEBUG("Read {} bytes from file {} for {} channels with {} samples per channel",
                       expected_size,
                       file_path_.get(),
                       num_channels,
                       samples_per_channel_.get());

    // Emit the buffer
    op_output.emit(buffer_info, "buffer");
  }

 private:
  Parameter<std::string> file_path_;
  Parameter<std::string> device_name_;
  Parameter<std::vector<std::string>> channel_names_;
  Parameter<std::vector<bool>> channel_outputs_;
  Parameter<size_t> samples_per_channel_;
  Parameter<bool> is_cyclic_;
  Parameter<size_t> sample_size_bytes_;

  size_t current_offset_ = 0;
  size_t file_size_ = 0;
};

// =============================================================================
// IIO Channel Conversion Operator
// =============================================================================

class IIOChannelConvertOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(IIOChannelConvertOp);
  IIOChannelConvertOp() = default;
  ~IIOChannelConvertOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<iio_buffer_info_t>("buffer_in");
    spec.output<iio_buffer_info_t>("buffer_out");
    spec.param(convert_channels_,
               "convert_channels",
               "Convert channels",
               "Apply IIO channel conversion",
               true);
  }

  void initialize() override {
    Operator::initialize();

    // Store IIO context and channels for conversion
    iio_context_ = nullptr;
    iio_device_ = nullptr;
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto buffer_info = op_input.receive<std::shared_ptr<iio_buffer_info_t>>("buffer_in").value();

    if (!buffer_info || !buffer_info->buffer) {
      HOLOSCAN_LOG_ERROR("IIOChannelConvertOp: Invalid buffer received");
      return;
    }

    // Create output buffer info (copy input structure)
    auto output_buffer_info = std::make_shared<iio_buffer_info_t>();
    output_buffer_info->samples_count = buffer_info->samples_count;
    output_buffer_info->is_cyclic = buffer_info->is_cyclic;
    output_buffer_info->device_name = buffer_info->device_name;
    output_buffer_info->enabled_channels = buffer_info->enabled_channels;

    // Get buffer properties
    const size_t num_channels = buffer_info->enabled_channels.size();
    const size_t samples_per_channel = buffer_info->samples_count;
    const size_t total_samples = samples_per_channel * num_channels;

    // Allocate output buffer
    output_buffer_info->buffer = new int16_t[total_samples];

    if (convert_channels_.get()) {
      // Apply IIO channel conversion
      convertChannelData(buffer_info, output_buffer_info, num_channels, samples_per_channel);
    } else {
      // Just copy data without conversion
      std::memcpy(output_buffer_info->buffer, buffer_info->buffer, total_samples * sizeof(int16_t));
    }

    // Emit the converted buffer
    op_output.emit(output_buffer_info, "buffer_out");
  }

 private:
  void convertChannelData(std::shared_ptr<iio_buffer_info_t> input_buffer,
                          std::shared_ptr<iio_buffer_info_t> output_buffer, size_t num_channels,
                          size_t samples_per_channel) {
    // Get or create IIO context if needed
    if (!iio_context_) {
      iio_context_ = iio_create_context_from_uri(G_URI);
      if (!iio_context_) {
        HOLOSCAN_LOG_ERROR("Failed to create IIO context for channel conversion");
        return;
      }
    }

    // Get device
    if (!iio_device_) {
      iio_device_ = iio_context_find_device(iio_context_, input_buffer->device_name.c_str());
      if (!iio_device_) {
        HOLOSCAN_LOG_ERROR("Failed to find IIO device: {}", input_buffer->device_name);
        return;
      }
    }

    // Get IIO channels for conversion
    std::vector<iio_channel*> iio_channels;
    for (const auto& ch_info : input_buffer->enabled_channels) {
      iio_channel* ch =
          iio_device_find_channel(iio_device_, ch_info.name.c_str(), ch_info.is_output);
      if (ch) {
        iio_channels.push_back(ch);
      } else {
        HOLOSCAN_LOG_WARN("Could not find IIO channel: {}", ch_info.name);
      }
    }

    const int16_t* input_data = static_cast<const int16_t*>(input_buffer->buffer);
    int16_t* output_data = static_cast<int16_t*>(output_buffer->buffer);

    if (num_channels == 1) {
      // Single channel conversion
      if (!iio_channels.empty()) {
        iio_channel* ch = iio_channels[0];

        for (size_t i = 0; i < samples_per_channel; ++i) {
          // Apply IIO channel conversion (handles scaling, offset, etc.)
          iio_channel_convert(ch, &output_data[i], &input_data[i]);
        }

        HOLOSCAN_LOG_DEBUG("Applied IIO channel conversion for single channel: {}",
                           input_buffer->enabled_channels[0].name);
      } else {
        // Fallback: copy without conversion
        std::memcpy(output_data, input_data, samples_per_channel * sizeof(int16_t));
      }
    } else {
      // Multi-channel conversion with interleaved data
      for (size_t sample = 0; sample < samples_per_channel; ++sample) {
        for (size_t ch = 0; ch < num_channels && ch < iio_channels.size(); ++ch) {
          size_t idx = sample * num_channels + ch;

          // Apply IIO channel conversion for each channel
          iio_channel_convert(iio_channels[ch], &output_data[idx], &input_data[idx]);
        }
      }

      HOLOSCAN_LOG_DEBUG("Applied IIO channel conversion for {} channels", num_channels);
    }
  }

  Parameter<bool> convert_channels_;
  iio_context* iio_context_;
  iio_device* iio_device_;
};

// =============================================================================
// CUDA Tensor Conversion Operator
// =============================================================================

class IIOBuffer2CudaTensorOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(IIOBuffer2CudaTensorOp);
  IIOBuffer2CudaTensorOp() = default;
  ~IIOBuffer2CudaTensorOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.input<iio_buffer_info_t>("buffer");
    spec.output<std::tuple<tensor_t<complex, 2>, cudaStream_t>>("tensor");
    spec.param(
        num_channels_, "num_channels", "Number of channels", "Number of channels in the data", 1U);
    spec.param(samples_per_channel_,
               "samples_per_channel",
               "Samples per channel",
               "Number of samples per channel",
               8192UL);
    spec.param(
        burst_size_, "burst_size", "Burst size", "Number of samples per burst for FFT", 1024);
    spec.param(num_bursts_, "num_bursts", "Number of bursts", "Number of bursts for FFT", 8);
    spec.param(adc_bits_, "adc_bits", "ADC bits", "ADC resolution in bits", 11);
  }

  void initialize() override {
    Operator::initialize();

    // Create CUDA stream
    cudaStreamCreate(&stream_);

    // Pre-allocate output tensor with shape matching FFT expectations
    // For FFT operator: (num_bursts, burst_size)
    make_tensor(output_tensor_,
                {static_cast<index_t>(num_bursts_.get()), static_cast<index_t>(burst_size_.get())});
  }

  void compute(InputContext& op_input, OutputContext& op_output, ExecutionContext&) override {
    auto buffer_info = op_input.receive<std::shared_ptr<iio_buffer_info_t>>("buffer").value();

    if (!buffer_info || !buffer_info->buffer) {
      HOLOSCAN_LOG_ERROR("IIOBuffer2CudaTensorOp: Invalid buffer received");
      return;
    }

    // Get buffer properties
    const int16_t* samples = static_cast<const int16_t*>(buffer_info->buffer);
    const size_t num_channels = buffer_info->enabled_channels.size();
    const size_t samples_per_channel = buffer_info->samples_count;

    // Validate dimensions
    if (num_channels != num_channels_.get() || samples_per_channel != samples_per_channel_.get()) {
      HOLOSCAN_LOG_WARN("Buffer dimensions mismatch: expected {}x{}, got {}x{}",
                        num_channels_.get(),
                        samples_per_channel_.get(),
                        num_channels,
                        samples_per_channel);
    }

    convertInterleavedIQToComplex(samples, samples_per_channel * 2);

    // Emit the tensor with stream
    op_output.emit(std::make_tuple(output_tensor_, stream_), "tensor");
  }

 private:
  void convertInterleavedIQToComplex(const int16_t* samples, size_t num_samples) {
    // Use proper ADC scaling based on bit depth (like GNU Radio)
    // Scale by the actual ADC resolution, not the int16 container size
    float adc_scale = 1.0f / (1 << adc_bits_.get());

    // Create temporary host buffer
    size_t total_complex_samples = num_samples / 2;  // I/Q pairs to complex
    std::vector<complex> host_data(total_complex_samples);

    // Convert interleaved I/Q samples to complex with proper ADC scaling
    for (size_t i = 0; i < total_complex_samples; ++i) {
      // Apply ADC scaling instead of full-scale int16 scaling
      float real = static_cast<float>(samples[i * 2]) * adc_scale;
      float imag = static_cast<float>(samples[i * 2 + 1]) * adc_scale;
      host_data[i] = complex(real, imag);
    }

    // Copy all data to GPU at once
    cudaMemcpyAsync(output_tensor_.Data(),
                    host_data.data(),
                    total_complex_samples * sizeof(complex),
                    cudaMemcpyHostToDevice,
                    stream_);
  }

  Parameter<unsigned int> num_channels_;
  Parameter<size_t> samples_per_channel_;
  Parameter<int> burst_size_;
  Parameter<int> num_bursts_;
  Parameter<int> adc_bits_;

  cudaStream_t stream_;
  tensor_t<complex, 2> output_tensor_;
};

// =============================================================================
// FFT Visualization Operators
// =============================================================================

// Common FFT processing utilities
namespace fft_utils {
inline std::vector<float> convertToMagnitudeSpectrum(const complex* host_data,
                                                     size_t burst_size, float power_offset,
                                                     float noise_floor_threshold = 1e-10f) {
  std::vector<float> magnitude_spectrum(burst_size);
  const float fft_normalization =
      1.0f / (static_cast<float>(burst_size) * static_cast<float>(burst_size));

  for (size_t i = 0; i < burst_size; ++i) {
    // Calculate magnitude squared
    const float real = host_data[i].real();
    const float imag = host_data[i].imag();
    const float mag_squared = real * real + imag * imag;

    // Apply FFT size normalization
    const float normalized_power = mag_squared * fft_normalization;

    // Apply logarithmic scale with smoother noise floor handling
    float db_value;
    if (normalized_power > noise_floor_threshold) {
      db_value = 10.0f * std::log10(normalized_power);
    } else if (normalized_power > 0) {
      // Use actual value instead of hard floor to avoid discontinuities
      db_value = 10.0f * std::log10(noise_floor_threshold);
    } else {
      // Only use -160 dB for actual zeros
      db_value = -160.0f;
    }
    magnitude_spectrum[i] = db_value + power_offset;
  }
  return magnitude_spectrum;
}

inline std::pair<float, float> findPeakFrequency(const std::vector<float>& magnitude_spectrum,
                                                 size_t burst_size, float freq_step_mhz) {
  auto peak_it = std::max_element(magnitude_spectrum.begin(), magnitude_spectrum.end());
  ssize_t peak_bin = std::distance(magnitude_spectrum.begin(), peak_it);
  float peak_freq_mhz =
      (static_cast<float>(peak_bin) - static_cast<float>(burst_size) / 2.0f) * freq_step_mhz;
  return {peak_freq_mhz, *peak_it};
}
}  // namespace fft_utils

class FFTGnuplotOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FFTGnuplotOp);
  FFTGnuplotOp() = default;
  ~FFTGnuplotOp() {
    if (pinned_host_data_) {
      cudaFreeHost(pinned_host_data_);
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.input<std::tuple<tensor_t<complex, 2>, cudaStream_t>>("buffer");
    spec.param(output_file_,
               "output_file",
               "Output file",
               "Base name for output files",
               std::string("fft_spectrum"));
    spec.param(selected_burst_, "selected_burst", "Selected burst", "Which burst to plot", 0);
    spec.param(
        max_frequency_, "max_frequency", "Max frequency", "Maximum frequency (Hz)", 1000000.0f);
    spec.param(power_offset_, "power_offset", "Power offset", "Power offset in dB", 0.0f);
    spec.param(adc_bits_, "adc_bits", "ADC bits", "ADC resolution in bits", 12);
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    auto tensor_data =
        op_input.receive<std::tuple<tensor_t<complex, 2>, cudaStream_t>>("buffer").value();
    auto& tensor = std::get<0>(tensor_data);
    auto stream = std::get<1>(tensor_data);

    // Synchronize stream to ensure data is ready
    cudaStreamSynchronize(stream);

    // Get tensor dimensions
    size_t num_bursts = tensor.Size(0);
    size_t burst_size = tensor.Size(1);

    // Select which burst to visualize
    size_t selected_burst = std::min(static_cast<size_t>(selected_burst_.get()), num_bursts - 1);

    HOLOSCAN_LOG_DEBUG(
        "Generating gnuplot for burst {} with {} FFT bins", selected_burst, burst_size);

    // Allocate pinned memory if needed or if size changed
    if (!pinned_host_data_ || pinned_buffer_size_ < burst_size) {
      if (pinned_host_data_) {
        cudaFreeHost(pinned_host_data_);
      }
      cudaMallocHost(&pinned_host_data_, burst_size * sizeof(complex));
      pinned_buffer_size_ = burst_size;
    }

    // Copy data for selected burst to pinned host memory (faster DMA transfer)
    cudaMemcpy(pinned_host_data_,
               tensor.Data() + selected_burst * burst_size,
               burst_size * sizeof(complex),
               cudaMemcpyDeviceToHost);

    // Convert to magnitude spectrum using pinned memory directly (no extra copy)
    auto magnitude_spectrum =
        fft_utils::convertToMagnitudeSpectrum(pinned_host_data_, burst_size, power_offset_.get());

    // Write data file for gnuplot with frequency axis from -fs/2 to +fs/2
    std::string data_file = output_file_.get() + ".dat";
    std::ofstream data_stream(data_file);

    float freq_step = max_frequency_.get() / static_cast<float>(burst_size);
    float freq_step_mhz = freq_step / 1e6f;  // Convert to MHz

    for (size_t i = 0; i < burst_size; ++i) {
      // Map frequency axis to [-fs/2, +fs/2] range in MHz
      float frequency_mhz =
          (static_cast<float>(i) - static_cast<float>(burst_size) / 2.0f) * freq_step_mhz;
      data_stream << frequency_mhz << " " << magnitude_spectrum[i] << std::endl;
    }
    data_stream.close();

    // Find peak frequency for title using common utility function
    auto [peak_freq_mhz, peak_magnitude] =
        fft_utils::findPeakFrequency(magnitude_spectrum, burst_size, freq_step_mhz);

    // Create gnuplot script
    std::string script_file = output_file_.get() + ".gp";
    std::ofstream script_stream(script_file);

    script_stream << "set terminal png size 1200,800\n";
    script_stream << "set output '" << output_file_.get() << ".png'\n";
    script_stream << "set title 'Pluto SDR FFT Spectrum - Peak: " << std::fixed
                  << std::setprecision(2) << peak_freq_mhz << " MHz (" << std::setprecision(1)
                  << peak_magnitude << " dB)'\n";
    script_stream << "set xlabel 'Frequency (MHz)'\n";
    script_stream << "set ylabel 'Magnitude (dB)'\n";

    script_stream << "set grid\n";
    script_stream << "set style line 1 linecolor rgb '#00ff00' linewidth 2\n";
    script_stream << "plot '" << data_file << "' with lines linestyle 1 title 'FFT Magnitude'\n";
    script_stream.close();

    // Execute gnuplot
    std::string gnuplot_cmd = "gnuplot " + script_file;
    int result = std::system(gnuplot_cmd.c_str());

    if (result == 0) {
      HOLOSCAN_LOG_INFO("Gnuplot spectrum saved to: {}.png", output_file_.get());
      HOLOSCAN_LOG_INFO(
          "Peak frequency: {:.2f} MHz with magnitude: {:.1f} dB", peak_freq_mhz, peak_magnitude);
    } else {
      HOLOSCAN_LOG_ERROR("Gnuplot execution failed with return code: {}", result);
    }

    // Display the plot using system image viewer (optional)
    std::string display_cmd = "xdg-open " + output_file_.get() + ".png &";
    std::system(display_cmd.c_str());

    // Clean up temporary files
    std::remove(data_file.c_str());
    std::remove(script_file.c_str());

    HOLOSCAN_LOG_INFO("FFT gnuplot visualization complete.");
  }

 private:
  Parameter<std::string> output_file_;
  Parameter<int> selected_burst_;
  Parameter<float> max_frequency_;
  Parameter<float> power_offset_;
  Parameter<int> adc_bits_;
  
  complex* pinned_host_data_ = nullptr;
  size_t pinned_buffer_size_ = 0;
};

class FFTGnuplotRealtimeOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(FFTGnuplotRealtimeOp);
  FFTGnuplotRealtimeOp() = default;
  ~FFTGnuplotRealtimeOp() {
    // Clean up gnuplot process
    if (gnuplot_pipe_) {
      pclose(gnuplot_pipe_);
    }
    // Clean up pinned memory
    if (pinned_host_data_) {
      cudaFreeHost(pinned_host_data_);
    }
  }

  void setup(OperatorSpec& spec) override {
    spec.input<std::tuple<tensor_t<complex, 2>, cudaStream_t>>("buffer");
    spec.param(
        max_frequency_, "max_frequency", "Max frequency", "Maximum frequency (Hz)", 1000000.0f);
    spec.param(power_offset_, "power_offset", "Power offset", "Power offset in dB", 0.0f);
    spec.param(adc_bits_, "adc_bits", "ADC bits", "ADC resolution in bits", 12);
    spec.param(update_interval_,
               "update_interval",
               "Update interval",
               "Update interval in milliseconds",
               100);
    spec.param(y_range_,
               "y_range",
               "Y-axis range",
               "Y-axis range [min, max]",
               std::vector<float>{-160.0f, 0.0f});
  }

  void initialize() override {
    Operator::initialize();

    // Open persistent gnuplot pipe
    gnuplot_pipe_ = popen("gnuplot", "w");
    if (!gnuplot_pipe_) {
      HOLOSCAN_LOG_ERROR("Failed to open gnuplot pipe");
      return;
    }

    // Initialize gnuplot for real-time plotting
    fprintf(gnuplot_pipe_, "set terminal x11 noraise\n");
    fprintf(gnuplot_pipe_, "set title 'Pluto SDR Real-Time FFT Spectrum'\n");
    fprintf(gnuplot_pipe_, "set xlabel 'Frequency (MHz)'\n");

    fprintf(gnuplot_pipe_, "set ylabel 'Magnitude (dB)'\n");
    fprintf(gnuplot_pipe_, "set yrange [%f:%f]\n", y_range_.get()[0], y_range_.get()[1]);

    fprintf(gnuplot_pipe_, "set grid\n");
    fprintf(gnuplot_pipe_, "set style line 1 linecolor rgb '#00ff00' linewidth 2\n");
    fflush(gnuplot_pipe_);

    // Initialize timing
    last_update_time_ = std::chrono::steady_clock::now();
  }

  void compute(InputContext& op_input, OutputContext&, ExecutionContext&) override {
    // Check update interval
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(current_time - last_update_time_)
            .count();

    if (elapsed < update_interval_.get()) {
      return;  // Skip this update
    }
    last_update_time_ = current_time;

    auto tensor_data =
        op_input.receive<std::tuple<tensor_t<complex, 2>, cudaStream_t>>("buffer").value();
    auto& tensor = std::get<0>(tensor_data);
    auto stream = std::get<1>(tensor_data);

    // Synchronize stream to ensure data is ready
    cudaStreamSynchronize(stream);

    // Get tensor dimensions
    size_t num_bursts = tensor.Size(0);
    size_t burst_size = tensor.Size(1);

    // Select first burst for real-time display
    size_t selected_burst = 0;

    // Allocate pinned memory if needed or if size changed
    if (!pinned_host_data_ || pinned_buffer_size_ < burst_size) {
      if (pinned_host_data_) {
        cudaFreeHost(pinned_host_data_);
      }
      cudaMallocHost(&pinned_host_data_, burst_size * sizeof(complex));
      pinned_buffer_size_ = burst_size;
    }

    // Copy data for selected burst to pinned host memory (faster DMA transfer)
    cudaMemcpy(pinned_host_data_,
               tensor.Data() + selected_burst * burst_size,
               burst_size * sizeof(complex),
               cudaMemcpyDeviceToHost);

    // Convert to magnitude spectrum using pinned memory directly (no extra copy)
    auto magnitude_spectrum =
        fft_utils::convertToMagnitudeSpectrum(pinned_host_data_, burst_size, power_offset_.get(), 1e-20f);

    // Find peak frequency for title
    float freq_step = max_frequency_.get() / static_cast<float>(burst_size);
    float freq_step_mhz = freq_step / 1e6f;
    auto [peak_freq_mhz, peak_magnitude] =
        fft_utils::findPeakFrequency(magnitude_spectrum, burst_size, freq_step_mhz);

    // Send data to gnuplot with updated title showing peak
    fprintf(gnuplot_pipe_,
            "set title 'Pluto SDR Real-Time FFT - Peak: %.2f MHz (%.1f dB)'\n",
            peak_freq_mhz,
            peak_magnitude);
    fprintf(gnuplot_pipe_, "plot '-' with lines linestyle 1 title 'FFT Magnitude'\n");

    for (size_t i = 0; i < burst_size; ++i) {
      // Map frequency axis to [-fs/2, +fs/2] range in MHz
      float frequency_mhz =
          (static_cast<float>(i) - static_cast<float>(burst_size) / 2.0f) * freq_step_mhz;
      fprintf(gnuplot_pipe_, "%f %f\n", frequency_mhz, magnitude_spectrum[i]);
    }
    fprintf(gnuplot_pipe_, "e\n");
    fflush(gnuplot_pipe_);
  }

 private:
  Parameter<float> max_frequency_;
  Parameter<float> power_offset_;
  Parameter<int> adc_bits_;
  Parameter<int> update_interval_;
  Parameter<std::vector<float>> y_range_;

  FILE* gnuplot_pipe_ = nullptr;
  std::chrono::steady_clock::time_point last_update_time_;
  
  // Pinned memory for faster GPU-to-host transfers
  complex* pinned_host_data_ = nullptr;
  size_t pinned_buffer_size_ = 0;
};

// =============================================================================
// Start Operator - Empty operator for dynamic flow control routing
// =============================================================================

class StartOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(StartOp);
  StartOp() = default;
  ~StartOp() = default;

  void setup(OperatorSpec& spec) override {
    // No inputs or outputs - pure router
  }

  void compute(InputContext&, OutputContext&, ExecutionContext&) override {
    // Empty compute - just triggers dynamic flow routing
  }
};

}  // namespace holoscan::ops
