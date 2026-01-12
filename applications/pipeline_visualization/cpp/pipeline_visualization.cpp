/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/**
 * @file pipeline_visualization.cpp
 * @brief Holoscan application demonstrating data generation and processing.
 *
 * This example showcases a Holoscan pipeline that:
 * 1. Generates a synthetic sine wave signal with time-varying frequency (10-20 Hz)
 * 2. Adds high-frequency modulation (300 Hz) to simulate measurement noise
 * 3. Processes the resulting signal through a sink operator
 * 4. Optionally streams data to a NATS server for external visualization
 *
 * The application demonstrates tensor manipulation, operator chaining, and data logging
 * capabilities in the Holoscan SDK.
 */

#include <getopt.h>

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>

#include <holoscan/holoscan.hpp>

#include "nats_logger.hpp"

namespace holoscan::ops {

/**
 * @brief Source operator that generates a sine wave signal.
 *
 * This operator produces a synthetic time-series signal consisting of a sine wave
 * with a gradually increasing frequency (10-20 Hz). It outputs 3000 samples per compute cycle.
 */
class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp)

  SourceOp() = default;

  void initialize() override {
    // Create an allocator for the operator
    allocator_ = fragment()->make_resource<UnboundedAllocator>("pool");
    // Add the allocator to the operator so that it is initialized
    add_arg(allocator_);

    // Call the base class initialize function
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override { spec.output<gxf::Entity>("out"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Generate a sine wave signal with time-varying frequency
    const uint32_t samples = 3000;  // Number of samples in the signal
    const float duration = 1;       // Duration of signal in seconds
    const float sample_time = duration / samples;
    const float omega = 2 * M_PI * frequency_;  // Angular frequency
    std::vector<float> wave(samples);
    for (uint32_t i = 0; i < samples; i++) { wave[i] = std::sin(omega * i * sample_time); }

    // Gradually increase frequency from 10 to 20 Hz, then wrap back to 10 Hz
    frequency_ += 0.1f;
    if (frequency_ > 20.f) {
      frequency_ = 10.f;
    }

    // Create a GXF tensor to hold the signal data
    auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();

    // Get handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    if (!allocator) {
      HOLOSCAN_LOG_ERROR("Failed to acquire allocator for SourceOp output tensor");
      return;
    }
    // Reshape the tensor to match the signal dimensions (samples x 1)
    auto reshape_result = gxf_tensor->reshape<float>(nvidia::gxf::Shape({samples, 1}),
                                                     nvidia::gxf::MemoryStorageType::kSystem,
                                                     allocator.value());
    if (!reshape_result) {
      HOLOSCAN_LOG_ERROR("Failed to reshape SourceOp output tensor");
      return;
    }

    // Copy the generated wave data to the tensor
    std::memcpy(gxf_tensor->pointer(), wave.data(), samples * sizeof(float));

    // Convert GXF tensor to Holoscan tensor format
    auto maybe_dl_ctx = (*gxf_tensor).toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
      HOLOSCAN_LOG_ERROR(
          "failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
      return;
    }
    auto holoscan_tensor = std::make_shared<Tensor>(maybe_dl_ctx.value());

    // Emit the tensor to the output port
    op_output.emit(holoscan_tensor, "out");
  }

 private:
  std::shared_ptr<UnboundedAllocator> allocator_;
  float frequency_ = 10.f;  // Current frequency of the sine wave in Hz
};

/**
 * @brief Modulation operator that adds high-frequency noise to the input signal.
 *
 * This operator receives a time-series signal and adds a 300 Hz sinusoidal modulation
 * with small amplitude (0.05) to simulate measurement noise or signal perturbation.
 */
class ModulateOp : public Operator {
 public:
  ModulateOp() = default;

  void initialize() override {
    // Create an allocator for the operator
    allocator_ = fragment()->make_resource<UnboundedAllocator>("pool");
    // Add the allocator to the operator so that it is initialized
    add_arg(allocator_);

    // Call the base class initialize function
    Operator::initialize();
  }

  void setup(OperatorSpec& spec) override {
    spec.input<gxf::Entity>("in");
    spec.output<gxf::Entity>("out");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Receive the input tensor from the source operator
    auto input_tensor = op_input.receive<std::shared_ptr<Tensor>>("in").value();
    const uint32_t samples = input_tensor->shape()[0];
    float* source_data = static_cast<float*>(input_tensor->data());

    // Create a new tensor for the modulated output
    auto gxf_tensor = std::make_shared<nvidia::gxf::Tensor>();

    // Get handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(context.context(),
                                                                         allocator_->gxf_cid());
    if (!allocator) {
      HOLOSCAN_LOG_ERROR("Failed to acquire allocator for ModulateOp output tensor");
      return;
    }
    // Reshape the tensor to match the input dimensions
    auto reshape_result = gxf_tensor->reshape<float>(nvidia::gxf::Shape({int32_t(samples), 1}),
                                                     nvidia::gxf::MemoryStorageType::kSystem,
                                                     allocator.value());
    if (!reshape_result) {
      HOLOSCAN_LOG_ERROR("Failed to reshape ModulateOp output tensor");
      return;
    }

    // Parameters for high-frequency modulation/noise
    const float frequency = 300.f;  // Modulation frequency in Hz
    const float amplitude = 0.05f;  // Amplitude of the modulation
    const float duration = 1;       // Duration of signal in seconds
    const float sample_time = duration / samples;
    const float omega = 2 * M_PI * frequency;  // Angular frequency

    // Add modulation to the input signal
    float* data = reinterpret_cast<float*>(gxf_tensor->pointer());
    for (uint32_t i = 0; i < samples; i++) {
      data[i] = source_data[i] + amplitude * std::sin(omega * i * sample_time);
    }

    // Convert GXF tensor to Holoscan tensor format
    auto maybe_dl_ctx = (*gxf_tensor).toDLManagedTensorContext();
    if (!maybe_dl_ctx) {
      HOLOSCAN_LOG_ERROR(
          "failed to get std::shared_ptr<DLManagedTensorContext> from nvidia::gxf::Tensor");
      return;
    }
    auto holoscan_tensor = std::make_shared<Tensor>(maybe_dl_ctx.value());

    // Emit the modulated signal to the output port
    op_output.emit(holoscan_tensor, "out");
  }

 private:
  std::shared_ptr<UnboundedAllocator> allocator_;
};

/**
 * @brief Sink operator that consumes the processed signal.
 *
 * This operator acts as the terminal node in the pipeline, receiving the modulated
 * signal. In this example, it simply receives the data without additional processing,
 * but could be extended to perform analysis or visualization.
 */
class SinkOp : public Operator {
 public:
  SinkOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<gxf::Entity>("in"); }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    // Receive the modulated signal from the previous operator
    auto tensor = op_input.receive<std::shared_ptr<Tensor>>("in").value();
    // Note: In this example, we simply receive the tensor. Additional processing
    // or visualization could be added here.
  }
};

}  // namespace holoscan::ops

/**
 * @brief Main application class for pipeline visualization demo.
 *
 * This application creates a pipeline that generates a time-varying sine wave,
 * adds high-frequency modulation to it, and processes the result. The application
 * can optionally log data to a NATS server for external monitoring and visualization.
 *
 * Pipeline flow: SourceOp -> ModulateOp -> SinkOp
 */
class TimeSeriesApp : public holoscan::Application {
 public:
  /**
   * @brief Constructor for TimeSeriesApp.
   *
   * @param disable_logger Whether to disable the NATS logger
   * @param nats_url URL for the NATS server connection
   * @param subject_prefix Prefix for NATS subject names
   * @param publish_rate Rate at which to publish data to NATS (Hz)
   */
  TimeSeriesApp(bool disable_logger, const std::string& nats_url, const std::string& subject_prefix,
                float publish_rate)
      : disable_logger_(disable_logger),
        nats_url_(nats_url),
        subject_prefix_(subject_prefix),
        publish_rate_(publish_rate) {}

  void compose() override {
    if (!disable_logger_) {
      // Create and configure the NATS logger for data streaming
      auto nats_logger = make_resource<holoscan::data_loggers::NatsLogger>(
          "nats_logger",
          holoscan::Arg("nats_url", nats_url_),
          holoscan::Arg("subject_prefix", subject_prefix_),
          holoscan::Arg("publish_rate", publish_rate_),
          from_config("nats_logger"));

      // Register the logger with the application
      add_data_logger(nats_logger);
    }

    // Create the three operators in the pipeline
    auto source_op = make_operator<holoscan::ops::SourceOp>(
        "source",
        // Limit the rate of the source operator
        make_condition<holoscan::PeriodicCondition>(
            "periodic-condition", holoscan::Arg("recess_period") = std::string("20hz")));
    auto modulate_op = make_operator<holoscan::ops::ModulateOp>("modulate");
    auto sink_op = make_operator<holoscan::ops::SinkOp>("sink");

    // Connect the operators: source -> modulate -> sink
    add_flow(source_op, modulate_op, {{"out", "in"}});
    add_flow(modulate_op, sink_op, {{"out", "in"}});
  }

 private:
  bool disable_logger_;         // Flag to disable NATS logging
  std::string nats_url_;        // NATS server URL
  std::string subject_prefix_;  // Prefix for NATS subjects
  float publish_rate_;          // Rate at which to publish data (Hz)
};

/**
 * @brief Main entry point for the pipeline visualization application.
 *
 * Parses command-line arguments, configures the application, and runs the pipeline.
 * Supports options for NATS server configuration and data logging control.
 *
 * @param argc Number of command-line arguments
 * @param argv Array of command-line argument strings
 * @return EXIT_SUCCESS (0) on successful execution
 */
int main(int argc, char** argv) {
  // Define command-line options
  // NOLINTBEGIN(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)
  struct option long_options[] = {{"help", no_argument, nullptr, 'h'},
                                  {"disable_logger", no_argument, nullptr, 'd'},
                                  {"config", required_argument, nullptr, 'c'},
                                  {"nats_url", required_argument, nullptr, 'u'},
                                  {"subject_prefix", required_argument, nullptr, 'p'},
                                  {"publish_rate", required_argument, nullptr, 'r'},
                                  {nullptr, 0, nullptr, 0}};
  // NOLINTEND(cppcoreguidelines-avoid-c-arrays,hicpp-avoid-c-arrays,modernize-avoid-c-arrays)

  // Set default values for configuration parameters
  bool disable_logger = false;
  std::string config_path;
  std::string nats_url("nats://0.0.0.0:4222");
  std::string subject_prefix("nats_demo");
  float publish_rate = 5.0f;

  // Parse command-line arguments
  while (true) {
    int option_index = 0;
    // NOLINTBEGIN(concurrency-mt-unsafe)
    const int c =
        getopt_long(argc, argv, "hdc:u:p:r:", static_cast<option*>(long_options), &option_index);
    // NOLINTEND(concurrency-mt-unsafe)

    // Break when no more options to parse
    if (c == -1) {
      break;
    }

    const std::string argument(optarg != nullptr ? optarg : "");
    switch (c) {
      case 'h':
      case '?':
        // Display help information
        std::cout << "Usage: " << argv[0] << " [options]" << std::endl
                  << "Options:" << std::endl
                  << "  -h, --help            display this information" << std::endl
                  << "  -d, --disable_logger  disable logger" << std::endl
                  << "  -c, --config          config file path" << std::endl
                  << "  -u, --nats_url        NATS URL (default: `" << nats_url << "`)" << std::endl
                  << "  -p, --subject_prefix  NATS subject prefix (default: `" << subject_prefix
                  << "`)" << std::endl
                  << "  -r, --publish_rate    publish rate (default: `" << publish_rate << "`)"
                  << std::endl;
        return EXIT_SUCCESS;

      case 'd':
        // Disable NATS logger
        disable_logger = true;
        break;
      case 'c':
        // Set custom config file path
        config_path = argument;
        break;
      case 'u':
        // Set NATS server URL
        nats_url = argument;
        break;
      case 'p':
        // Set NATS subject prefix
        subject_prefix = argument;
        break;
      case 'r':
        // Set publish rate
        try {
          publish_rate = std::stof(argument);
        } catch (const std::invalid_argument& e) {
          HOLOSCAN_LOG_ERROR("Invalid publish rate: `{}`: {}", argument, e.what());
        } catch (const std::out_of_range& e) {
          HOLOSCAN_LOG_ERROR("Publish rate out of range: `{}`: {}", argument, e.what());
        }
        break;
      default:
        throw std::runtime_error(fmt::format("Unhandled option `{}`", static_cast<char>(c)));
    }
  }

  // Create the application instance with parsed parameters
  auto app = holoscan::make_application<TimeSeriesApp>(
      disable_logger, nats_url, subject_prefix, publish_rate);

  // Load configuration from YAML file
  // If no config path specified, use default file in the same directory as executable
  if (config_path.empty()) {
    std::filesystem::path config_dir;
    try {
      config_dir = std::filesystem::canonical(argv[0]).parent_path();
    } catch (const std::filesystem::filesystem_error&) {
      config_dir = std::filesystem::current_path();
    }
    config_path = config_dir / "pipeline_visualization.yaml";
  }
  app->config(config_path);

  // Execute the application pipeline
  app->run();

  return 0;
}
